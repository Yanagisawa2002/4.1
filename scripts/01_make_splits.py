from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Dict, List, Mapping, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.checks import (  # noqa: E402
    assert_cross_disease_disjointness,
    assert_cross_drug_disjointness,
    assert_edge_disjointness,
    assert_ho_alignment_with_kg,
)
from src.splits import derive_ho_splits, split_kg_indications  # noqa: E402

Edge = Tuple[str, str]
HOQuad = Tuple[str, str, str, str]
SPLIT_NAMES = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create KG train/val/test splits and derive HO splits from KG."
    )
    parser.add_argument(
        "--kg-positive",
        required=True,
        help=(
            "Path to KG indication positives. Supports either "
            "drug,disease or relation,x_id,x_type,y_id,y_type schema."
        ),
    )
    parser.add_argument(
        "--ho",
        required=True,
        help=(
            "Path to HO table. Supports either "
            "drug,protein,pathway,disease or "
            "drugbank_id,protein_id,pathway_id,disease_id schema."
        ),
    )
    parser.add_argument(
        "--split-type",
        required=True,
        choices=["random", "cross-drug", "cross-disease"],
        help="Split strategy for KG positives.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Fixed random seed.")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument(
        "--out-dir",
        default="outputs/splits",
        help="Directory where split files are written.",
    )
    return parser.parse_args()


def _guess_delimiter(path: Path) -> str:
    sample = path.read_text(encoding="utf-8-sig")[:4096]
    try:
        return csv.Sniffer().sniff(sample, delimiters=",\t;").delimiter
    except csv.Error:
        return ","


def _read_rows(path: Path) -> List[dict]:
    delimiter = _guess_delimiter(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if not reader.fieldnames:
            raise ValueError(f"No header detected in file: {path}")
        rows: List[dict] = []
        for row in reader:
            clean: Dict[str, str] = {}
            for key, value in row.items():
                if key is None:
                    continue
                clean[key.strip()] = value.strip() if isinstance(value, str) else value
            rows.append(clean)
    if not rows:
        raise ValueError(f"No data rows found in {path}")
    return rows


def _read_kg_positive_edges(path: Path) -> List[Edge]:
    rows = _read_rows(path)
    first_row = rows[0]

    drug_col = _resolve_column(first_row, ("drug",), required=False)
    disease_col = _resolve_column(first_row, ("disease",), required=False)
    if drug_col and disease_col:
        edges: List[Edge] = []
        for i, row in enumerate(rows, start=2):
            drug = _normalize_drug_id(row[drug_col])
            disease = row[disease_col].strip()
            if not drug or not disease:
                raise ValueError(f"Empty drug/disease at {path}:{i}")
            edges.append((drug, disease))
        return edges

    rel_col = _resolve_column(first_row, ("relation", "rel", "edge_type"), required=False)
    src_col = _resolve_column(first_row, ("x_id", "src", "source", "head", "u"))
    dst_col = _resolve_column(first_row, ("y_id", "dst", "target", "tail", "v"))
    src_type_col = _resolve_column(first_row, ("x_type", "src_type", "source_type"), required=False)
    dst_type_col = _resolve_column(first_row, ("y_type", "dst_type", "target_type"), required=False)

    edges: List[Edge] = []
    for i, row in enumerate(rows, start=2):
        relation = row[rel_col].strip() if rel_col else "indication"
        if relation != "indication":
            continue

        src = row[src_col].strip()
        dst = row[dst_col].strip()
        if not src or not dst:
            raise ValueError(f"Empty x_id/y_id at {path}:{i}")
        src_type = row[src_type_col].strip() if src_type_col else None
        dst_type = row[dst_type_col].strip() if dst_type_col else None

        pair = _extract_drug_disease_pair(src, dst, src_type, dst_type)
        if pair is None:
            raise ValueError(
                "Unable to infer drug-disease indication pair at "
                f"{path}:{i} from row={(src, relation, dst, src_type, dst_type)}"
            )
        edges.append(pair)

    if not edges:
        raise ValueError(f"No indication edges found in {path}")
    return edges


def _read_ho_quads(path: Path) -> List[HOQuad]:
    rows = _read_rows(path)
    first_row = rows[0]
    drug_col = _resolve_column(first_row, ("drug", "drug_id", "drugbank_id", "x_id"))
    protein_col = _resolve_column(first_row, ("protein", "protein_id", "target_id"))
    pathway_col = _resolve_column(first_row, ("pathway", "pathway_id"))
    disease_col = _resolve_column(first_row, ("disease", "disease_id", "diseaseid"))

    quads: List[HOQuad] = []
    for i, row in enumerate(rows, start=2):
        quad = (
            _normalize_drug_id(row[drug_col]),
            row[protein_col].strip(),
            _normalize_pathway_id(row[pathway_col]),
            row[disease_col].strip(),
        )
        if any(not item for item in quad):
            raise ValueError(f"Empty HO value at {path}:{i}")
        quads.append(quad)
    return quads


def _resolve_column(
    row: Mapping[str, str],
    candidates: Sequence[str],
    required: bool = True,
) -> str | None:
    key_map = {key.lower(): key for key in row.keys()}
    for candidate in candidates:
        key = key_map.get(candidate.lower())
        if key is not None:
            return key
    if required:
        raise ValueError(f"Missing required column. Tried candidates={list(candidates)}")
    return None


def _extract_drug_disease_pair(
    src: str,
    dst: str,
    src_type: str | None,
    dst_type: str | None,
) -> Edge | None:
    if src_type is None or dst_type is None:
        return (_normalize_drug_id(src), dst)
    if src_type == "drug" and dst_type == "disease":
        return (_normalize_drug_id(src), dst)
    if src_type == "disease" and dst_type == "drug":
        return (_normalize_drug_id(dst), src)
    return None


def _normalize_drug_id(drug_id: str) -> str:
    value = drug_id.strip()
    if value.startswith("drug::"):
        return value
    if value.upper().startswith("DB"):
        return f"drug::{value}"
    return value


def _normalize_pathway_id(pathway_id: str) -> str:
    value = pathway_id.strip()
    if value.startswith("pathway::"):
        return value
    if "reactome:" in value:
        reactome_id = value.split("reactome:", 1)[1]
        if reactome_id.startswith("R-HSA-"):
            return f"pathway::{reactome_id}"
    if value.startswith("R-HSA-"):
        return f"pathway::{value}"
    return value


def _write_csv(path: Path, header: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    kg_positive_path = Path(args.kg_positive)
    ho_path = Path(args.ho)
    out_dir = Path(args.out_dir)

    kg_positive = _read_kg_positive_edges(kg_positive_path)
    ho_quads = _read_ho_quads(ho_path)

    kg_split = split_kg_indications(
        positive_edges=kg_positive,
        split_type=args.split_type,
        seed=args.seed,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        negative_k=1,
    )
    ho_splits = derive_ho_splits(ho_quads, kg_split.pos)

    assert_edge_disjointness(kg_split.pos)
    if args.split_type == "cross-drug":
        assert_cross_drug_disjointness(kg_split.pos)
    if args.split_type == "cross-disease":
        assert_cross_disease_disjointness(kg_split.pos)
    assert_ho_alignment_with_kg(ho_splits, kg_split.pos)

    for split_name in SPLIT_NAMES:
        _write_csv(
            out_dir / f"kg_pos_{split_name}.csv",
            ("drug", "disease"),
            kg_split.pos[split_name],
        )
        _write_csv(
            out_dir / f"kg_neg_{split_name}.csv",
            ("drug", "disease"),
            kg_split.neg[split_name],
        )
        _write_csv(
            out_dir / f"ho_{split_name}.csv",
            ("drug", "protein", "pathway", "disease"),
            ho_splits[split_name],
        )

    metadata = {
        "split_type": args.split_type,
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "counts": {
            "kg_pos": {name: len(kg_split.pos[name]) for name in SPLIT_NAMES},
            "kg_neg": {name: len(kg_split.neg[name]) for name in SPLIT_NAMES},
            "ho": {name: len(ho_splits[name]) for name in SPLIT_NAMES},
        },
    }
    (out_dir / "split_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
