from __future__ import annotations

from itertools import combinations
from typing import Dict, Mapping, Sequence, Tuple

Edge = Tuple[str, str]
HOQuad = Tuple[str, str, str, str]
SPLIT_NAMES = ("train", "val", "test")


def assert_edge_disjointness(kg_pos_splits: Mapping[str, Sequence[Edge]]) -> None:
    _require_split_keys(kg_pos_splits)
    split_sets: Dict[str, set[Edge]] = {
        split_name: {(e[0], e[1]) for e in kg_pos_splits[split_name]}
        for split_name in SPLIT_NAMES
    }
    for left, right in combinations(SPLIT_NAMES, 2):
        overlap = split_sets[left] & split_sets[right]
        if overlap:
            example = next(iter(overlap))
            raise AssertionError(
                f"Edge leakage across splits ({left}, {right}). "
                f"Overlap count={len(overlap)}; example={example}"
            )


def assert_cross_drug_disjointness(kg_pos_splits: Mapping[str, Sequence[Edge]]) -> None:
    _require_split_keys(kg_pos_splits)
    train_drugs = {drug for drug, _ in kg_pos_splits["train"]}
    test_drugs = {drug for drug, _ in kg_pos_splits["test"]}
    overlap = train_drugs & test_drugs
    if overlap:
        example = next(iter(overlap))
        raise AssertionError(
            "Cross-drug constraint violated: train and test drugs overlap. "
            f"Overlap count={len(overlap)}; example={example}"
        )


def assert_cross_disease_disjointness(kg_pos_splits: Mapping[str, Sequence[Edge]]) -> None:
    _require_split_keys(kg_pos_splits)
    train_diseases = {disease for _, disease in kg_pos_splits["train"]}
    test_diseases = {disease for _, disease in kg_pos_splits["test"]}
    overlap = train_diseases & test_diseases
    if overlap:
        example = next(iter(overlap))
        raise AssertionError(
            "Cross-disease constraint violated: train and test diseases overlap. "
            f"Overlap count={len(overlap)}; example={example}"
        )


def assert_ho_alignment_with_kg(
    ho_splits: Mapping[str, Sequence[HOQuad]],
    kg_pos_splits: Mapping[str, Sequence[Edge]],
) -> None:
    _require_split_keys(ho_splits)
    _require_split_keys(kg_pos_splits)

    pair_to_split: Dict[Edge, str] = {}
    for split_name in SPLIT_NAMES:
        for drug, disease in kg_pos_splits[split_name]:
            pair = (drug, disease)
            if pair in pair_to_split and pair_to_split[pair] != split_name:
                raise AssertionError(
                    f"KG pair appears in multiple splits: {pair} -> "
                    f"{pair_to_split[pair]}, {split_name}"
                )
            pair_to_split[pair] = split_name

    for split_name in SPLIT_NAMES:
        for quad in ho_splits[split_name]:
            if len(quad) != 4:
                raise AssertionError(f"Invalid HO quadruplet (need 4 values): {quad}")
            pair = (quad[0], quad[3])
            mapped_split = pair_to_split.get(pair)
            if mapped_split is None:
                raise AssertionError(
                    f"HO quadruplet uses non-positive KG pair: {pair}"
                )
            if mapped_split != split_name:
                raise AssertionError(
                    "HO split leakage: quadruplet pair mapped to different KG split. "
                    f"pair={pair}, ho_split={split_name}, kg_split={mapped_split}"
                )


def assert_ho_alignment(
    ho_train: Sequence[HOQuad],
    kg_pos_train: Sequence[Edge],
) -> None:
    kg_pairs = {(drug, disease) for drug, disease in kg_pos_train}
    for quad in ho_train:
        if len(quad) != 4:
            raise AssertionError(f"Invalid HO quadruplet (need 4 values): {quad}")
        pair = (quad[0], quad[3])
        if pair not in kg_pairs:
            raise AssertionError(
                "HO train alignment violated: quadruplet pair not in KG train positives. "
                f"pair={pair}"
            )


def assert_ho_train_only(split_names: Sequence[str]) -> None:
    invalid = [name for name in split_names if name != "train"]
    if invalid:
        raise AssertionError(
            "HO leakage guard violated: training must use only ho_train. "
            f"Found disallowed splits={invalid}"
        )


def assert_pair_loader_integrity(
    kg_pos_splits: Mapping[str, Sequence[Edge]],
    kg_neg_splits: Mapping[str, Sequence[Edge]],
) -> None:
    _require_split_keys(kg_pos_splits)
    _require_split_keys(kg_neg_splits)

    for split_name in SPLIT_NAMES:
        n_pos = len(kg_pos_splits[split_name])
        n_neg = len(kg_neg_splits[split_name])
        if n_pos != n_neg:
            raise AssertionError(
                "K=1 constraint violated in materialized split files: "
                f"split={split_name}, positives={n_pos}, negatives={n_neg}"
            )

    all_positive: set[Edge] = set()
    for split_name in SPLIT_NAMES:
        all_positive.update((drug, disease) for drug, disease in kg_pos_splits[split_name])

    for split_name in SPLIT_NAMES:
        neg_set = {(drug, disease) for drug, disease in kg_neg_splits[split_name]}
        overlap = neg_set & all_positive
        if overlap:
            example = next(iter(overlap))
            raise AssertionError(
                "Negative edges overlap known positives. "
                f"split={split_name}, overlap_count={len(overlap)}, example={example}"
            )

    pair_to_label: Dict[Edge, int] = {}
    for split_name in SPLIT_NAMES:
        for pair in kg_pos_splits[split_name]:
            normalized = (pair[0], pair[1])
            existing = pair_to_label.get(normalized)
            if existing is not None and existing != 1:
                raise AssertionError(
                    "Conflicting labels across splits for pair "
                    f"{normalized}: seen negative and positive."
                )
            pair_to_label[normalized] = 1

        for pair in kg_neg_splits[split_name]:
            normalized = (pair[0], pair[1])
            existing = pair_to_label.get(normalized)
            if existing is not None and existing != 0:
                raise AssertionError(
                    "Conflicting labels across splits for pair "
                    f"{normalized}: seen positive and negative."
                )
            pair_to_label[normalized] = 0


def _require_split_keys(mapping: Mapping[str, Sequence]) -> None:
    missing = [k for k in SPLIT_NAMES if k not in mapping]
    if missing:
        raise ValueError(f"Missing split keys: {missing}. Required: {SPLIT_NAMES}")
