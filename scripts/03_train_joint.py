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

try:
    from tqdm.auto import tqdm

    TQDM_AVAILABLE = True
except ModuleNotFoundError:
    tqdm = None
    TQDM_AVAILABLE = False

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.checks import assert_ho_train_only  # noqa: E402
from src.data import BaseDataBundle, HOTrainLoader, prepare_base_data  # noqa: E402
from src.loss_ho import ho_component_infonce_loss  # noqa: E402
from src.metrics import binary_metrics_from_logits  # noqa: E402
from src.model_rgcn import BaseRGCNPairModel, ModelConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train joint Base+HO R-GCN model with pair BCE + HO InfoNCE."
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
            "Directory containing kg_pos_*.csv, kg_neg_*.csv, and ho_*.csv from split script. "
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
    parser.add_argument("--ho-batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--pair-hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--lambda-ho", type=float, default=0.1)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument(
        "--balance-key",
        default="drug",
        choices=["drug", "disease"],
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--no-tqdm",
        action="store_true",
        help="Disable tqdm progress bars.",
    )
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


def _to_ho_batch(
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    drug_index, protein_index, pathway_index, disease_index, weight = batch
    return {
        "drug_index": drug_index.to(device),
        "protein_index": protein_index.to(device),
        "pathway_index": pathway_index.to(device),
        "disease_index": disease_index.to(device),
        "weight": weight.to(device),
    }


def train_one_epoch(
    model: BaseRGCNPairModel,
    bundle: BaseDataBundle,
    train_loader: torch.utils.data.DataLoader,
    ho_train_loader: HOTrainLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    lambda_ho: float,
    tau: float,
    use_tqdm: bool,
) -> Dict[str, float]:
    model.train()
    edge_index = bundle.graph.edge_index.to(device)
    edge_type = bundle.graph.edge_type.to(device)

    total_loss = 0.0
    total_pair_loss = 0.0
    total_ho_loss = 0.0
    total_pair_count = 0
    total_ho_count = 0
    total_wq = 0.0

    ho_iter = iter(ho_train_loader)
    batch_iter = train_loader
    if use_tqdm:
        batch_iter = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{total_epochs} train",
            leave=False,
            dynamic_ncols=True,
        )

    for pair_batch in batch_iter:
        drug_index, disease_index, labels = pair_batch
        drug_index = drug_index.to(device)
        disease_index = disease_index.to(device)
        labels = labels.to(device)

        try:
            ho_batch = next(ho_iter)
        except StopIteration:
            ho_iter = iter(ho_train_loader)
            ho_batch = next(ho_iter)
        ho_batch_dict = _to_ho_batch(ho_batch, device=device)

        optimizer.zero_grad(set_to_none=True)
        logits, ho_outputs = model(
            edge_index=edge_index,
            edge_type=edge_type,
            drug_index=drug_index,
            disease_index=disease_index,
            ho_batch=ho_batch_dict,
        )
        pair_loss = F.binary_cross_entropy_with_logits(logits, labels)
        ho_loss = ho_component_infonce_loss(ho_outputs=ho_outputs, tau=tau)
        loss = pair_loss + (lambda_ho * ho_loss)
        loss.backward()
        optimizer.step()

        pair_batch_size = labels.size(0)
        ho_batch_size = ho_batch_dict["weight"].size(0)
        total_loss += float(loss.item()) * pair_batch_size
        total_pair_loss += float(pair_loss.item()) * pair_batch_size
        total_ho_loss += float(ho_loss.item()) * ho_batch_size
        total_pair_count += int(pair_batch_size)
        total_ho_count += int(ho_batch_size)
        total_wq += float(ho_batch_dict["weight"].sum().item())

        if use_tqdm:
            batch_iter.set_postfix(
                loss=f"{(total_loss / max(total_pair_count, 1)):.4f}",
                pair=f"{(total_pair_loss / max(total_pair_count, 1)):.4f}",
                ho=f"{(total_ho_loss / max(total_ho_count, 1)):.4f}",
            )

    if total_pair_count == 0:
        raise ValueError("Train pair loader is empty.")
    if total_ho_count == 0:
        raise ValueError("HO train loader is empty.")

    return {
        "loss": total_loss / total_pair_count,
        "pair_loss": total_pair_loss / total_pair_count,
        "ho_loss": total_ho_loss / total_ho_count,
        "mean_wq": total_wq / total_ho_count,
    }


@torch.no_grad()
def evaluate(
    model: BaseRGCNPairModel,
    bundle: BaseDataBundle,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    desc: str,
    use_tqdm: bool,
) -> Dict[str, float]:
    model.eval()
    edge_index = bundle.graph.edge_index.to(device)
    edge_type = bundle.graph.edge_type.to(device)

    total_loss = 0.0
    total_count = 0
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    batch_iter = data_loader
    if use_tqdm:
        batch_iter = tqdm(
            data_loader,
            desc=desc,
            leave=False,
            dynamic_ncols=True,
        )

    for drug_index, disease_index, labels in batch_iter:
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
        if use_tqdm:
            batch_iter.set_postfix(loss=f"{(total_loss / max(total_count, 1)):.4f}")

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

    ho_counts = {split_name: split.total for split_name, split in bundle.ho_splits.items()}
    return {
        "graph": {
            "num_nodes": bundle.graph.num_nodes,
            "num_edges": int(bundle.graph.edge_index.size(1)),
            "num_relations": bundle.graph.num_relations,
            "num_nodes_by_type": bundle.graph.num_nodes_by_type,
        },
        "pairs": pair_counts,
        "ho": ho_counts,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    use_tqdm = (not args.no_tqdm) and TQDM_AVAILABLE
    if (not args.no_tqdm) and (not TQDM_AVAILABLE):
        print(
            "tqdm is not installed; running without progress bars. "
            "Install it with: pip install tqdm (or pip install -r requirements.txt)."
        )

    bundle = prepare_base_data(
        node_types_path=args.node_types,
        kg_edges_path=args.kg_edges,
        split_dir=args.split_dir,
        split_type=args.split_type,
        indication_relation=args.indication_relation,
        keep_only_train_indication=True,
    )

    assert_ho_train_only(("train",))

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
    ho_train_loader = bundle.make_ho_train_loader(
        batch_size=args.ho_batch_size,
        seed=args.seed,
        balance_key=args.balance_key,
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

    epoch_iter = range(1, args.epochs + 1)
    if use_tqdm:
        epoch_iter = tqdm(
            epoch_iter,
            desc="Training epochs",
            dynamic_ncols=True,
        )

    for epoch in epoch_iter:
        ho_train_loader.set_epoch(epoch - 1)
        train_stats = train_one_epoch(
            model=model,
            bundle=bundle,
            train_loader=train_loader,
            ho_train_loader=ho_train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            total_epochs=args.epochs,
            lambda_ho=args.lambda_ho,
            tau=args.tau,
            use_tqdm=use_tqdm,
        )
        val_metrics = evaluate(
            model=model,
            bundle=bundle,
            data_loader=val_loader,
            device=device,
            desc=f"Epoch {epoch}/{args.epochs} val",
            use_tqdm=use_tqdm,
        )

        history_item = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "pair_loss": train_stats["pair_loss"],
            "ho_loss": train_stats["ho_loss"],
            "mean_wq": train_stats["mean_wq"],
            "ho_batch_size": float(args.ho_batch_size),
            "lambda_ho": float(args.lambda_ho),
            "tau": float(args.tau),
            "val_loss": val_metrics["loss"],
            "val_auroc": val_metrics["auroc"],
            "val_auprc": val_metrics["auprc"],
        }
        history.append(history_item)
        if use_tqdm:
            tqdm.write(json.dumps(history_item))
            epoch_iter.set_postfix(
                train_loss=f"{train_stats['loss']:.4f}",
                val_auprc=f"{val_metrics['auprc']:.4f}",
            )
        else:
            print(json.dumps(history_item))

        if val_metrics["auprc"] > best_val_auprc:
            best_val_auprc = val_metrics["auprc"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    final_val = evaluate(
        model=model,
        bundle=bundle,
        data_loader=val_loader,
        device=device,
        desc="Final val",
        use_tqdm=use_tqdm,
    )
    final_test = evaluate(
        model=model,
        bundle=bundle,
        data_loader=test_loader,
        device=device,
        desc="Final test",
        use_tqdm=use_tqdm,
    )

    summary = {
        "seed": args.seed,
        "split_type": args.split_type,
        "indication_relation": args.indication_relation,
        "best_epoch": best_epoch,
        "lambda_ho": args.lambda_ho,
        "tau": args.tau,
        "ho_batch_size": args.ho_batch_size,
        "balance_key": args.balance_key,
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
