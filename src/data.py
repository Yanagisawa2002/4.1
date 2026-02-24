from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import csv
import math
from pathlib import Path
import random
from typing import Dict, Iterable, Iterator, List, Mapping, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Sampler, TensorDataset

from src.checks import (
    assert_cross_disease_disjointness,
    assert_cross_drug_disjointness,
    assert_edge_disjointness,
    assert_ho_alignment,
    assert_ho_train_only,
    assert_pair_loader_integrity,
)

Edge = Tuple[str, str]
HOQuad = Tuple[str, str, str, str]
CanonicalEdgeType = Tuple[str, str, str]
SPLIT_NAMES = ("train", "val", "test")
PREFERRED_NODE_TYPE_ORDER = ("drug", "disease", "gene/protein", "protein", "pathway")


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
class HOSplitData:
    quadruplets: Tuple[HOQuad, ...]
    drug_index: torch.LongTensor
    protein_index: torch.LongTensor
    pathway_index: torch.LongTensor
    disease_index: torch.LongTensor
    weight: torch.FloatTensor

    @property
    def total(self) -> int:
        return len(self.quadruplets)


class _BalancedHOBatchSampler(Sampler[List[int]]):
    def __init__(
        self,
        key_indices: Sequence[int],
        batch_size: int,
        seed: int,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0 for HO sampling")
        if len(key_indices) == 0:
            raise ValueError("Cannot create HO sampler from empty ho_train.")

        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0
        self.num_samples = len(key_indices)

        grouped: Dict[int, List[int]] = defaultdict(list)
        for idx, key in enumerate(key_indices):
            grouped[int(key)].append(idx)
        self.groups = {group: tuple(indices) for group, indices in grouped.items()}
        self.group_keys = tuple(sorted(self.groups.keys()))
        if not self.group_keys:
            raise ValueError("No groups found for HO balanced sampling.")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed + self.epoch)
        per_group: Dict[int, List[int]] = {}
        for group, indices in self.groups.items():
            shuffled = list(indices)
            rng.shuffle(shuffled)
            per_group[group] = shuffled

        group_order = list(self.group_keys)
        rng.shuffle(group_order)
        if not group_order:
            raise ValueError("Balanced HO sampler has no groups to sample from.")

        pointers = {group: 0 for group in group_order}
        sampled_indices: List[int] = []
        group_cursor = 0
        while len(sampled_indices) < self.num_samples:
            group = group_order[group_cursor % len(group_order)]
            group_cursor += 1
            group_items = per_group[group]
            idx = group_items[pointers[group] % len(group_items)]
            pointers[group] += 1
            sampled_indices.append(idx)

        for start in range(0, len(sampled_indices), self.batch_size):
            yield sampled_indices[start : start + self.batch_size]

    def __len__(self) -> int:
        return math.ceil(self.num_samples / self.batch_size)


class HOTrainLoader:
    def __init__(
        self,
        dataset: TensorDataset,
        batch_sampler: _BalancedHOBatchSampler,
        num_workers: int = 0,
    ) -> None:
        self._batch_sampler = batch_sampler
        self._loader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
        )

    def set_epoch(self, epoch: int) -> None:
        self._batch_sampler.set_epoch(epoch)

    def __iter__(self):
        return iter(self._loader)

    def __len__(self) -> int:
        return len(self._loader)


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
    ho_splits: Dict[str, HOSplitData]

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

    def make_ho_train_loader(
        self,
        batch_size: int,
        seed: int,
        balance_key: str = "drug",
        num_workers: int = 0,
    ) -> HOTrainLoader:
        assert_ho_train_only(("train",))
        if balance_key not in {"drug", "disease"}:
            raise ValueError("balance_key must be one of {'drug', 'disease'}.")
        ho_train = self.ho_splits["train"]
        if ho_train.total == 0:
            raise ValueError("ho_train is empty; cannot create HO train loader.")

        if balance_key == "drug":
            key_tensor = ho_train.drug_index
        else:
            key_tensor = ho_train.disease_index
        batch_sampler = _BalancedHOBatchSampler(
            key_indices=key_tensor.tolist(),
            batch_size=batch_size,
            seed=seed,
        )
        dataset = TensorDataset(
            ho_train.drug_index,
            ho_train.protein_index,
            ho_train.pathway_index,
            ho_train.disease_index,
            ho_train.weight,
        )
        return HOTrainLoader(
            dataset=dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
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
    ho_splits = load_ho_splits(
        split_dir=split_dir,
        graph=graph,
        pair_splits=pair_splits,
    )
    return BaseDataBundle(
        graph=graph,
        pair_splits=pair_splits,
        ho_splits=ho_splits,
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
    kg_path = Path(kg_edges_path)
    if not kg_path.exists():
        raise ValueError(f"KG path does not exist: {kg_path}")

    file_paths: List[Path]
    strict_single_file = kg_path.is_file()
    if strict_single_file:
        file_paths = [kg_path]
    else:
        file_paths = sorted(kg_path.glob("*.csv"))
        if not file_paths:
            raise ValueError(f"No CSV files found under KG directory: {kg_path}")

    typed_edges: List[Tuple[str, str, str, str, str]] = []
    kept_indication_pairs: set[Edge] = set()
    recognized_files: List[str] = []

    for file_path in file_paths:
        edges_from_file, kept_pairs_from_file, recognized = _load_edges_from_file(
            path=file_path,
            node_type_map=node_type_map,
            indication_relation=indication_relation,
            train_positive_pairs=train_positive_pairs,
            keep_only_train_indication=keep_only_train_indication,
            strict=strict_single_file,
        )
        if not recognized:
            continue
        typed_edges.extend(edges_from_file)
        kept_indication_pairs.update(kept_pairs_from_file)
        recognized_files.append(file_path.name)

    if not recognized_files:
        raise ValueError(
            f"No supported KG edge files found for path={kg_path}. "
            "Supported schemas: relation,x_id,x_type,y_id,y_type and disease_id,pathway_id."
        )

    if keep_only_train_indication:
        missing = train_positive_pairs - kept_indication_pairs
        if missing:
            for drug, disease in sorted(missing):
                drug_type = node_type_map.get(drug)
                disease_type = node_type_map.get(disease)
                if drug_type != "drug" or disease_type != "disease":
                    raise ValueError(
                        "Cannot inject missing train indication edge because node types "
                        f"are invalid for pair={(drug, disease)} with "
                        f"types={(drug_type, disease_type)}"
                    )
                typed_edges.append((drug, indication_relation, disease, "drug", "disease"))

    if not typed_edges:
        raise ValueError("No KG edges available after loading/filtering.")
    return _dedupe_typed_edges(typed_edges)


def _load_edges_from_file(
    path: Path,
    node_type_map: Mapping[str, str],
    indication_relation: str,
    train_positive_pairs: set[Edge],
    keep_only_train_indication: bool,
    strict: bool,
) -> tuple[List[Tuple[str, str, str, str, str]], set[Edge], bool]:
    rows = _read_rows(path)
    first_row = rows[0]

    src_col = _resolve_column(first_row, ("x_id", "src", "source", "head", "u"), required=False)
    dst_col = _resolve_column(first_row, ("y_id", "dst", "target", "tail", "v"), required=False)
    rel_col = _resolve_column(first_row, ("relation", "rel", "edge_type", "predicate"), required=False)
    src_type_col = _resolve_column(
        first_row, ("x_type", "src_type", "source_type"), required=False
    )
    dst_type_col = _resolve_column(
        first_row, ("y_type", "dst_type", "target_type"), required=False
    )

    if src_col and dst_col:
        typed_edges: List[Tuple[str, str, str, str, str]] = []
        kept_pairs: set[Edge] = set()
        for i, row in enumerate(rows, start=2):
            src = row[src_col].strip()
            dst = row[dst_col].strip()
            relation = row[rel_col].strip() if rel_col else path.stem
            if not src or not relation or not dst:
                raise ValueError(f"Empty src/relation/dst value at row {i} in {path}")

            src_type = row[src_type_col].strip() if src_type_col else node_type_map.get(src)
            dst_type = row[dst_type_col].strip() if dst_type_col else node_type_map.get(dst)
            if src_type is None or dst_type is None:
                raise ValueError(
                    f"Missing node type mapping for edge row {i} ({src}, {relation}, {dst})."
                )
            if src not in node_type_map or dst not in node_type_map:
                raise ValueError(f"Edge uses node missing from node type file at row {i} in {path}")
            if node_type_map[src] != src_type or node_type_map[dst] != dst_type:
                raise ValueError(
                    f"Edge type mismatch at row {i} in {path}: "
                    f"({src}:{src_type}, {dst}:{dst_type}) does not match node type mapping."
                )

            if relation == indication_relation:
                pair = _extract_drug_disease_pair(src, src_type, dst, dst_type)
                if pair is None:
                    raise ValueError(
                        "Indication relation must connect drug and disease nodes. "
                        f"Found row {i} in {path}: ({src_type}, {relation}, {dst_type})"
                    )

                if keep_only_train_indication and pair not in train_positive_pairs:
                    continue
                kept_pairs.add(pair)
                src, dst = pair[0], pair[1]
                src_type, dst_type = "drug", "disease"

            typed_edges.append((src, relation, dst, src_type, dst_type))
        return typed_edges, kept_pairs, True

    disease_col = _resolve_column(first_row, ("disease_id",), required=False)
    pathway_col = _resolve_column(first_row, ("pathway_id",), required=False)
    if disease_col and pathway_col:
        typed_edges = []
        for i, row in enumerate(rows, start=2):
            disease = row[disease_col].strip()
            pathway = row[pathway_col].strip()
            if not disease or not pathway:
                raise ValueError(f"Empty disease/pathway at row {i} in {path}")
            if disease not in node_type_map or pathway not in node_type_map:
                raise ValueError(
                    f"Edge uses node missing from node mapping at row {i} in {path}"
                )
            if node_type_map[disease] != "disease" or node_type_map[pathway] != "pathway":
                raise ValueError(
                    f"Expected disease/pathway node types at row {i} in {path}, "
                    f"got ({node_type_map[disease]}, {node_type_map[pathway]})"
                )
            typed_edges.append((disease, "disease_pathway", pathway, "disease", "pathway"))
        return typed_edges, set(), True

    if strict:
        raise ValueError(
            f"Unsupported KG edge schema in file {path}. "
            "Supported schemas: relation,x_id,x_type,y_id,y_type and disease_id,pathway_id."
        )
    return [], set(), False


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


def load_ho_splits(
    split_dir: str | Path,
    graph: RGCNGraph,
    pair_splits: Mapping[str, PairSplitData],
) -> Dict[str, HOSplitData]:
    split_dir = Path(split_dir)
    ho_quads_by_split: Dict[str, List[HOQuad]] = {}
    for split_name in SPLIT_NAMES:
        ho_quads_by_split[split_name] = _read_ho_quad_file(split_dir / f"ho_{split_name}.csv")

    assert_ho_alignment(
        ho_train=ho_quads_by_split["train"],
        kg_pos_train=pair_splits["train"].positive_edges,
    )

    protein_freq: Dict[str, int] = defaultdict(int)
    pathway_freq: Dict[str, int] = defaultdict(int)
    for _, protein, pathway, _ in ho_quads_by_split["train"]:
        protein_freq[protein] += 1
        pathway_freq[pathway] += 1

    split_data: Dict[str, HOSplitData] = {}
    for split_name in SPLIT_NAMES:
        quads = tuple(ho_quads_by_split[split_name])
        drug_indices: List[int] = []
        protein_indices: List[int] = []
        pathway_indices: List[int] = []
        disease_indices: List[int] = []
        weights: List[float] = []

        for quad in quads:
            drug, protein, pathway, disease = quad
            drug_indices.append(
                _resolve_global_node_index(
                    graph=graph,
                    node_id=drug,
                    allowed_types=("drug",),
                    context=f"ho_{split_name}",
                )
            )
            protein_indices.append(
                _resolve_global_node_index(
                    graph=graph,
                    node_id=protein,
                    allowed_types=("gene/protein", "protein"),
                    context=f"ho_{split_name}",
                )
            )
            pathway_indices.append(
                _resolve_global_node_index(
                    graph=graph,
                    node_id=pathway,
                    allowed_types=("pathway",),
                    context=f"ho_{split_name}",
                )
            )
            disease_indices.append(
                _resolve_global_node_index(
                    graph=graph,
                    node_id=disease,
                    allowed_types=("disease",),
                    context=f"ho_{split_name}",
                )
            )

            if split_name == "train":
                denom = protein_freq[protein] + pathway_freq[pathway]
                if denom <= 0:
                    raise ValueError(
                        "Invalid HO train frequency denominator for quad "
                        f"{quad}: protein_freq={protein_freq[protein]}, "
                        f"pathway_freq={pathway_freq[pathway]}"
                    )
                weights.append(1.0 / math.sqrt(float(denom)))
            else:
                weights.append(1.0)

        split_data[split_name] = HOSplitData(
            quadruplets=quads,
            drug_index=torch.tensor(drug_indices, dtype=torch.long),
            protein_index=torch.tensor(protein_indices, dtype=torch.long),
            pathway_index=torch.tensor(pathway_indices, dtype=torch.long),
            disease_index=torch.tensor(disease_indices, dtype=torch.long),
            weight=torch.tensor(weights, dtype=torch.float32),
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


def _read_ho_quad_file(path: Path) -> List[HOQuad]:
    quads: List[HOQuad] = []
    delimiter = _guess_delimiter(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if not reader.fieldnames:
            raise ValueError(f"No header detected in {path}")
        header_row = {field.strip(): "" for field in reader.fieldnames if field is not None}
        drug_col = _resolve_column(header_row, ("drug",))
        protein_col = _resolve_column(header_row, ("protein",))
        pathway_col = _resolve_column(header_row, ("pathway",))
        disease_col = _resolve_column(header_row, ("disease",))

        for i, row in enumerate(reader, start=2):
            drug = row[drug_col].strip() if isinstance(row[drug_col], str) else row[drug_col]
            protein = (
                row[protein_col].strip()
                if isinstance(row[protein_col], str)
                else row[protein_col]
            )
            pathway = (
                row[pathway_col].strip()
                if isinstance(row[pathway_col], str)
                else row[pathway_col]
            )
            disease = (
                row[disease_col].strip()
                if isinstance(row[disease_col], str)
                else row[disease_col]
            )
            if not drug or not protein or not pathway or not disease:
                raise ValueError(f"Empty HO value at row {i} in {path}")
            quads.append((drug, protein, pathway, disease))
    return quads


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


def _resolve_global_node_index(
    graph: RGCNGraph,
    node_id: str,
    allowed_types: Sequence[str],
    context: str,
) -> int:
    for node_type in allowed_types:
        typed_mapping = graph.node_to_local.get(node_type)
        if typed_mapping is None:
            continue
        local = typed_mapping.get(node_id)
        if local is None:
            continue
        return graph.node_offsets[node_type] + local
    raise ValueError(
        f"{context} references unknown node '{node_id}' for allowed types {tuple(allowed_types)}"
    )


def _ordered_node_types(types: Iterable[str]) -> List[str]:
    type_set = set(types)
    ordered: List[str] = [name for name in PREFERRED_NODE_TYPE_ORDER if name in type_set]
    ordered.extend(sorted(type_set - set(ordered)))
    return ordered


def _dedupe_typed_edges(
    edges: Sequence[Tuple[str, str, str, str, str]],
) -> List[Tuple[str, str, str, str, str]]:
    seen: set[Tuple[str, str, str, str, str]] = set()
    deduped: List[Tuple[str, str, str, str, str]] = []
    for edge in edges:
        if edge in seen:
            continue
        seen.add(edge)
        deduped.append(edge)
    return deduped


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
