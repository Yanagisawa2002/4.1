from __future__ import annotations

from dataclasses import dataclass
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.checks import (
    assert_cross_disease_disjointness,
    assert_cross_drug_disjointness,
    assert_edge_disjointness,
    assert_pair_loader_integrity,
)

Edge = Tuple[str, str]
CanonicalEdgeType = Tuple[str, str, str]
SPLIT_NAMES = ("train", "val", "test")
PREFERRED_NODE_TYPE_ORDER = ("drug", "disease", "protein", "pathway")


@dataclass(frozen=True)
class PairSplitData:
    drug_index: torch.LongTensor
    disease_index: torch.LongTensor
    labels: torch.FloatTensor
    positive_edges: Tuple[Edge, ...]
    negative_edges: Tuple[Edge, ...]

    @property
    def total(self) -> int:
        return int(self.labels.numel())

    @property
    def pos_count(self) -> int:
        return len(self.positive_edges)

    @property
    def neg_count(self) -> int:
        return len(self.negative_edges)


@dataclass(frozen=True)
class RGCNGraph:
    node_to_local: Dict[str, Dict[str, int]]
    local_to_node: Dict[str, Tuple[str, ...]]
    node_offsets: Dict[str, int]
    num_nodes_by_type: Dict[str, int]
    relation_to_id: Dict[CanonicalEdgeType, int]
    edge_index_by_type: Dict[CanonicalEdgeType, torch.LongTensor]
    edge_index: torch.LongTensor
    edge_type: torch.LongTensor

    @property
    def num_nodes(self) -> int:
        return sum(self.num_nodes_by_type.values())

    @property
    def num_relations(self) -> int:
        return len(self.relation_to_id)


@dataclass(frozen=True)
class BaseDataBundle:
    graph: RGCNGraph
    pair_splits: Dict[str, PairSplitData]
    ho_splits: Dict[str, None]

    def make_pair_loader(
        self,
        split_name: str,
        batch_size: int,
        shuffle: bool,
        seed: int,
        num_workers: int = 0,
    ) -> DataLoader:
        if split_name not in self.pair_splits:
            raise KeyError(f"Unknown split_name={split_name}")
        split = self.pair_splits[split_name]
        dataset = TensorDataset(split.drug_index, split.disease_index, split.labels)
        generator = torch.Generator()
        generator.manual_seed(seed)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            generator=generator,
        )


def prepare_base_data(
    node_types_path: str | Path,
    kg_edges_path: str | Path,
    split_dir: str | Path,
    split_type: str | None,
    indication_relation: str = "indication",
    keep_only_train_indication: bool = True,
) -> BaseDataBundle:
    split_dir = Path(split_dir)
    node_type_map = load_node_type_mapping(node_types_path)
    train_pos_pairs = set(_read_pair_edge_file(split_dir / "kg_pos_train.csv"))

    kg_edges = load_kg_edges(
        kg_edges_path=kg_edges_path,
        node_type_map=node_type_map,
        indication_relation=indication_relation,
        train_positive_pairs=train_pos_pairs,
        keep_only_train_indication=keep_only_train_indication,
    )
    graph = build_rgcn_graph(node_type_map=node_type_map, kg_edges=kg_edges)
    pair_splits = load_pair_splits(
        split_dir=split_dir,
        graph=graph,
        split_type=split_type,
    )
    return BaseDataBundle(
        graph=graph,
        pair_splits=pair_splits,
        # Reserved for future HO batching integration.
        ho_splits={name: None for name in SPLIT_NAMES},
    )


def load_node_type_mapping(
    node_types_path: str | Path,
) -> Dict[str, str]:
    rows = _read_rows(Path(node_types_path))
    node_id_col = _resolve_column(rows[0], ("node_id", "node", "id"))
    node_type_col = _resolve_column(rows[0], ("node_type", "type"))

    node_type_map: Dict[str, str] = {}
    for i, row in enumerate(rows, start=2):
        node_id = row[node_id_col]
        node_type = row[node_type_col]
        if not node_id or not node_type:
            raise ValueError(f"Empty node/type value at row {i} in {node_types_path}")
        if node_id in node_type_map and node_type_map[node_id] != node_type:
            raise ValueError(
                f"Conflicting node type mapping for node={node_id}: "
                f"{node_type_map[node_id]} vs {node_type}"
            )
        node_type_map[node_id] = node_type

    if not node_type_map:
        raise ValueError(f"No node mappings found in {node_types_path}")
    return node_type_map


def load_kg_edges(
    kg_edges_path: str | Path,
    node_type_map: Mapping[str, str],
    indication_relation: str,
    train_positive_pairs: set[Edge],
    keep_only_train_indication: bool = True,
) -> List[Tuple[str, str, str, str, str]]:
    rows = _read_rows(Path(kg_edges_path))
    src_col = _resolve_column(rows[0], ("src", "source", "head", "u"))
    rel_col = _resolve_column(rows[0], ("relation", "rel", "edge_type", "predicate"))
    dst_col = _resolve_column(rows[0], ("dst", "target", "tail", "v"))
    src_type_col = _resolve_column(rows[0], ("src_type", "source_type"), required=False)
    dst_type_col = _resolve_column(rows[0], ("dst_type", "target_type"), required=False)

    typed_edges: List[Tuple[str, str, str, str, str]] = []
    kept_indication_pairs: set[Edge] = set()

    for i, row in enumerate(rows, start=2):
        src = row[src_col]
        relation = row[rel_col]
        dst = row[dst_col]
        if not src or not relation or not dst:
            raise ValueError(f"Empty src/relation/dst value at row {i} in {kg_edges_path}")

        src_type = row[src_type_col] if src_type_col else node_type_map.get(src)
        dst_type = row[dst_type_col] if dst_type_col else node_type_map.get(dst)
        if src_type is None or dst_type is None:
            raise ValueError(
                f"Missing node type mapping for edge row {i} ({src}, {relation}, {dst})."
            )
        if src not in node_type_map or dst not in node_type_map:
            raise ValueError(f"Edge uses node missing from node type file at row {i}")
        if node_type_map[src] != src_type or node_type_map[dst] != dst_type:
            raise ValueError(
                f"Edge type mismatch at row {i}: ({src}:{src_type}, {dst}:{dst_type}) "
                "does not match node type mapping."
            )

        if relation == indication_relation and keep_only_train_indication:
            pair = _extract_drug_disease_pair(src, src_type, dst, dst_type)
            if pair is None:
                raise ValueError(
                    "Indication relation must connect drug and disease nodes. "
                    f"Found row {i}: ({src_type}, {relation}, {dst_type})"
                )
            if pair not in train_positive_pairs:
                continue
            kept_indication_pairs.add(pair)

        typed_edges.append((src, relation, dst, src_type, dst_type))

    if keep_only_train_indication:
        missing = train_positive_pairs - kept_indication_pairs
        if missing:
            example = next(iter(missing))
            raise ValueError(
                "KG does not contain all train indication positives. "
                f"Missing count={len(missing)}; example={example}"
            )

    if not typed_edges:
        raise ValueError("No KG edges available after loading/filtering.")
    return typed_edges


def build_rgcn_graph(
    node_type_map: Mapping[str, str],
    kg_edges: Sequence[Tuple[str, str, str, str, str]],
) -> RGCNGraph:
    nodes_by_type: Dict[str, List[str]] = {}
    for node_id, node_type in node_type_map.items():
        nodes_by_type.setdefault(node_type, []).append(node_id)
    for node_type in nodes_by_type:
        nodes_by_type[node_type].sort()

    ordered_types = _ordered_node_types(nodes_by_type.keys())
    node_to_local: Dict[str, Dict[str, int]] = {}
    local_to_node: Dict[str, Tuple[str, ...]] = {}
    num_nodes_by_type: Dict[str, int] = {}
    node_offsets: Dict[str, int] = {}

    offset = 0
    for node_type in ordered_types:
        typed_nodes = tuple(nodes_by_type[node_type])
        local_to_node[node_type] = typed_nodes
        node_to_local[node_type] = {node_id: i for i, node_id in enumerate(typed_nodes)}
        num_nodes_by_type[node_type] = len(typed_nodes)
        node_offsets[node_type] = offset
        offset += len(typed_nodes)

    edge_lists: Dict[CanonicalEdgeType, List[Tuple[int, int]]] = {}
    for src, relation, dst, src_type, dst_type in kg_edges:
        src_local = node_to_local[src_type][src]
        dst_local = node_to_local[dst_type][dst]
        edge_lists.setdefault((src_type, relation, dst_type), []).append((src_local, dst_local))

    if not edge_lists:
        raise ValueError("Graph has no edges.")

    relation_to_id: Dict[CanonicalEdgeType, int] = {}
    edge_index_by_type: Dict[CanonicalEdgeType, torch.LongTensor] = {}
    global_edge_index_parts: List[torch.LongTensor] = []
    global_edge_type_parts: List[torch.LongTensor] = []

    for rel_id, edge_type in enumerate(sorted(edge_lists.keys())):
        relation_to_id[edge_type] = rel_id
        pairs = edge_lists[edge_type]
        src_local = torch.tensor([p[0] for p in pairs], dtype=torch.long)
        dst_local = torch.tensor([p[1] for p in pairs], dtype=torch.long)
        local_edge_index = torch.stack((src_local, dst_local), dim=0)
        edge_index_by_type[edge_type] = local_edge_index

        src_type, _, dst_type = edge_type
        src_global = src_local + node_offsets[src_type]
        dst_global = dst_local + node_offsets[dst_type]
        global_edge_index_parts.append(torch.stack((src_global, dst_global), dim=0))
        global_edge_type_parts.append(torch.full((len(pairs),), rel_id, dtype=torch.long))

    edge_index = torch.cat(global_edge_index_parts, dim=1)
    edge_type = torch.cat(global_edge_type_parts, dim=0)

    return RGCNGraph(
        node_to_local=node_to_local,
        local_to_node=local_to_node,
        node_offsets=node_offsets,
        num_nodes_by_type=num_nodes_by_type,
        relation_to_id=relation_to_id,
        edge_index_by_type=edge_index_by_type,
        edge_index=edge_index,
        edge_type=edge_type,
    )


def load_pair_splits(
    split_dir: str | Path,
    graph: RGCNGraph,
    split_type: str | None,
) -> Dict[str, PairSplitData]:
    split_dir = Path(split_dir)
    kg_pos_splits: Dict[str, List[Edge]] = {}
    kg_neg_splits: Dict[str, List[Edge]] = {}
    for split_name in SPLIT_NAMES:
        kg_pos_splits[split_name] = _read_pair_edge_file(split_dir / f"kg_pos_{split_name}.csv")
        kg_neg_splits[split_name] = _read_pair_edge_file(split_dir / f"kg_neg_{split_name}.csv")

    assert_edge_disjointness(kg_pos_splits)
    if split_type == "cross-drug":
        assert_cross_drug_disjointness(kg_pos_splits)
    if split_type == "cross-disease":
        assert_cross_disease_disjointness(kg_pos_splits)
    assert_pair_loader_integrity(kg_pos_splits=kg_pos_splits, kg_neg_splits=kg_neg_splits)

    drug_to_local = graph.node_to_local.get("drug")
    disease_to_local = graph.node_to_local.get("disease")
    if drug_to_local is None or disease_to_local is None:
        raise ValueError("Graph must contain both 'drug' and 'disease' node types.")
    drug_offset = graph.node_offsets["drug"]
    disease_offset = graph.node_offsets["disease"]

    split_data: Dict[str, PairSplitData] = {}
    for split_name in SPLIT_NAMES:
        pos_edges = tuple(kg_pos_splits[split_name])
        neg_edges = tuple(kg_neg_splits[split_name])
        combined_edges = list(pos_edges) + list(neg_edges)
        labels = [1.0] * len(pos_edges) + [0.0] * len(neg_edges)

        drug_global_ids: List[int] = []
        disease_global_ids: List[int] = []
        for drug, disease in combined_edges:
            if drug not in drug_to_local:
                raise ValueError(
                    f"Pair split references unknown drug node '{drug}' in split={split_name}"
                )
            if disease not in disease_to_local:
                raise ValueError(
                    f"Pair split references unknown disease node '{disease}' in split={split_name}"
                )
            drug_global_ids.append(drug_offset + drug_to_local[drug])
            disease_global_ids.append(disease_offset + disease_to_local[disease])

        split_data[split_name] = PairSplitData(
            drug_index=torch.tensor(drug_global_ids, dtype=torch.long),
            disease_index=torch.tensor(disease_global_ids, dtype=torch.long),
            labels=torch.tensor(labels, dtype=torch.float32),
            positive_edges=pos_edges,
            negative_edges=neg_edges,
        )

    return split_data


def _read_pair_edge_file(path: Path) -> List[Edge]:
    rows = _read_rows(path)
    drug_col = _resolve_column(rows[0], ("drug",))
    disease_col = _resolve_column(rows[0], ("disease",))
    edges: List[Edge] = []
    for i, row in enumerate(rows, start=2):
        drug = row[drug_col]
        disease = row[disease_col]
        if not drug or not disease:
            raise ValueError(f"Empty drug/disease at row {i} in {path}")
        edges.append((drug, disease))
    return edges


def _extract_drug_disease_pair(
    src: str,
    src_type: str,
    dst: str,
    dst_type: str,
) -> Edge | None:
    if src_type == "drug" and dst_type == "disease":
        return (src, dst)
    if src_type == "disease" and dst_type == "drug":
        return (dst, src)
    return None


def _ordered_node_types(types: Iterable[str]) -> List[str]:
    type_set = set(types)
    ordered: List[str] = [name for name in PREFERRED_NODE_TYPE_ORDER if name in type_set]
    ordered.extend(sorted(type_set - set(ordered)))
    return ordered


def _read_rows(path: Path) -> List[Dict[str, str]]:
    delimiter = _guess_delimiter(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if not reader.fieldnames:
            raise ValueError(f"No header detected in {path}")
        rows: List[Dict[str, str]] = []
        for row in reader:
            clean_row: Dict[str, str] = {}
            for key, value in row.items():
                if key is None:
                    continue
                clean_row[key.strip()] = value.strip() if isinstance(value, str) else value
            rows.append(clean_row)
    if not rows:
        raise ValueError(f"No data rows found in {path}")
    return rows


def _guess_delimiter(path: Path) -> str:
    sample = path.read_text(encoding="utf-8-sig")[:4096]
    try:
        return csv.Sniffer().sniff(sample, delimiters=",\t;").delimiter
    except csv.Error:
        return ","


def _resolve_column(
    row: Mapping[str, str],
    candidates: Sequence[str],
    required: bool = True,
) -> str | None:
    key_map = {key.lower(): key for key in row.keys()}
    for candidate in candidates:
        if candidate.lower() in key_map:
            return key_map[candidate.lower()]
    if required:
        raise ValueError(f"Missing required column. Tried candidates={candidates}")
    return None
