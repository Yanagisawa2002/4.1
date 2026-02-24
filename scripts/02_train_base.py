from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import random
import sys
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import BaseDataBundle, prepare_base_data  # noqa: E402
from src.metrics import binary_metrics_from_logits  # noqa: E402
from src.model_rgcn import BaseRGCNPairModel, ModelConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Base R-GCN model with L_pair only (no HO loss)."
    )
    parser.add_argument(
        "--kg-edges",
        default="data/KG",
        help=(
            "KG edges CSV/TSV path or a directory containing KG CSV files. "
            "Current project default: data/KG"
        ),
    )
    parser.add_argument(
        "--node-types",
        default="data/KG/nodes.csv",
        help=(
            "Node-type mapping CSV/TSV path. Supports id,type (current project) "
            "and node_id,node_type. Default: data/KG/nodes.csv"
        ),
    )
    parser.add_argument(
        "--split-dir",
        default="outputs/splits/random",
        help=(
            "Directory containing kg_pos_*.csv and kg_neg_*.csv from split script. "
            "Default: outputs/splits/random"
        ),
    )
    parser.add_argument(
        "--split-type",
        default="random",
        choices=["random", "cross-drug", "cross-disease"],
        help="Split type for integrity checks. Default: random.",
    )
    parser.add_argument("--indication-relation", default="indication")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--pair-hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, or cuda[:index].",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to write final training summary JSON.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def train_one_epoch(
    model: BaseRGCNPairModel,
    bundle: BaseDataBundle,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    edge_index = bundle.graph.edge_index.to(device)
    edge_type = bundle.graph.edge_type.to(device)

    total_loss = 0.0
    total_count = 0
    for drug_index, disease_index, labels in train_loader:
        drug_index = drug_index.to(device)
        disease_index = disease_index.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(
            edge_index=edge_index,
            edge_type=edge_type,
            drug_index=drug_index,
            disease_index=disease_index,
            ho_batch=None,
        )
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_count += int(batch_size)

    if total_count == 0:
        raise ValueError("Train loader is empty.")
    return total_loss / total_count


@torch.no_grad()
def evaluate(
    model: BaseRGCNPairModel,
    bundle: BaseDataBundle,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    edge_index = bundle.graph.edge_index.to(device)
    edge_type = bundle.graph.edge_type.to(device)

    total_loss = 0.0
    total_count = 0
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    for drug_index, disease_index, labels in data_loader:
        drug_index = drug_index.to(device)
        disease_index = disease_index.to(device)
        labels = labels.to(device)

        logits = model(
            edge_index=edge_index,
            edge_type=edge_type,
            drug_index=drug_index,
            disease_index=disease_index,
            ho_batch=None,
        )
        loss = F.binary_cross_entropy_with_logits(logits, labels)

        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_count += int(batch_size)
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())

    if total_count == 0:
        raise ValueError("Eval loader is empty.")

    logits_tensor = torch.cat(all_logits, dim=0)
    labels_tensor = torch.cat(all_labels, dim=0)
    metrics = binary_metrics_from_logits(logits=logits_tensor, labels=labels_tensor)
    metrics["loss"] = total_loss / total_count
    return metrics


def summarize_counts(bundle: BaseDataBundle) -> Dict[str, object]:
    pair_counts = {}
    for split_name, split in bundle.pair_splits.items():
        pair_counts[split_name] = {
            "total": split.total,
            "pos": split.pos_count,
            "neg": split.neg_count,
        }

    return {
        "graph": {
            "num_nodes": bundle.graph.num_nodes,
            "num_edges": int(bundle.graph.edge_index.size(1)),
            "num_relations": bundle.graph.num_relations,
            "num_nodes_by_type": bundle.graph.num_nodes_by_type,
        },
        "pairs": pair_counts,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    bundle = prepare_base_data(
        node_types_path=args.node_types,
        kg_edges_path=args.kg_edges,
        split_dir=args.split_dir,
        split_type=args.split_type,
        indication_relation=args.indication_relation,
        keep_only_train_indication=True,
    )

    train_loader = bundle.make_pair_loader(
        split_name="train",
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    val_loader = bundle.make_pair_loader(
        split_name="val",
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.seed + 1,
        num_workers=args.num_workers,
    )
    test_loader = bundle.make_pair_loader(
        split_name="test",
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.seed + 2,
        num_workers=args.num_workers,
    )

    config = ModelConfig(
        num_nodes=bundle.graph.num_nodes,
        num_relations=bundle.graph.num_relations,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        pair_hidden_dim=args.pair_hidden_dim,
        dropout=args.dropout,
    )
    model = BaseRGCNPairModel(config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_epoch = 0
    best_val_auprc = float("-inf")
    best_state = copy.deepcopy(model.state_dict())
    history: list[Dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            bundle=bundle,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
        )
        val_metrics = evaluate(model=model, bundle=bundle, data_loader=val_loader, device=device)

        history_item = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_auroc": val_metrics["auroc"],
            "val_auprc": val_metrics["auprc"],
        }
        history.append(history_item)
        print(json.dumps(history_item))

        if val_metrics["auprc"] > best_val_auprc:
            best_val_auprc = val_metrics["auprc"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    final_val = evaluate(model=model, bundle=bundle, data_loader=val_loader, device=device)
    final_test = evaluate(model=model, bundle=bundle, data_loader=test_loader, device=device)

    summary = {
        "seed": args.seed,
        "split_type": args.split_type,
        "indication_relation": args.indication_relation,
        "best_epoch": best_epoch,
        "counts": summarize_counts(bundle),
        "val": final_val,
        "test": final_test,
    }
    print(json.dumps(summary, indent=2))

    if args.output_json is not None:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
