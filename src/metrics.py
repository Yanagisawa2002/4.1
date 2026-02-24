from __future__ import annotations

from typing import Dict

import torch


def binary_auroc_from_scores(
    labels: torch.Tensor,
    scores: torch.Tensor,
) -> float:
    labels, scores = _prepare_tensors(labels=labels, scores=scores)
    n_pos = int(labels.sum().item())
    n_neg = int((1.0 - labels).sum().item())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = torch.argsort(scores, descending=True)
    sorted_labels = labels[order]
    true_pos = torch.cumsum(sorted_labels, dim=0)
    false_pos = torch.cumsum(1.0 - sorted_labels, dim=0)

    tpr = true_pos / float(n_pos)
    fpr = false_pos / float(n_neg)

    zero = torch.tensor([0.0], dtype=tpr.dtype, device=tpr.device)
    tpr = torch.cat((zero, tpr), dim=0)
    fpr = torch.cat((zero, fpr), dim=0)
    return float(torch.trapz(tpr, fpr).item())


def binary_auprc_from_scores(
    labels: torch.Tensor,
    scores: torch.Tensor,
) -> float:
    labels, scores = _prepare_tensors(labels=labels, scores=scores)
    n_pos = int(labels.sum().item())
    if n_pos == 0:
        return float("nan")

    order = torch.argsort(scores, descending=True)
    sorted_labels = labels[order]
    true_pos = torch.cumsum(sorted_labels, dim=0)
    false_pos = torch.cumsum(1.0 - sorted_labels, dim=0)

    precision = true_pos / (true_pos + false_pos).clamp_min(1.0)
    recall = true_pos / float(n_pos)

    one = torch.tensor([1.0], dtype=precision.dtype, device=precision.device)
    zero = torch.tensor([0.0], dtype=recall.dtype, device=recall.device)
    precision = torch.cat((one, precision), dim=0)
    recall = torch.cat((zero, recall), dim=0)

    delta_recall = recall[1:] - recall[:-1]
    return float(torch.sum(delta_recall * precision[1:]).item())


def binary_metrics_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> Dict[str, float]:
    labels, logits = _prepare_tensors(labels=labels, scores=logits)
    probs = torch.sigmoid(logits)
    return {
        "auroc": binary_auroc_from_scores(labels=labels, scores=probs),
        "auprc": binary_auprc_from_scores(labels=labels, scores=probs),
    }


def _prepare_tensors(
    labels: torch.Tensor,
    scores: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if labels.numel() != scores.numel():
        raise ValueError(
            f"labels and scores must have same number of elements: "
            f"{labels.numel()} vs {scores.numel()}"
        )
    labels = labels.reshape(-1).to(dtype=torch.float64)
    scores = scores.reshape(-1).to(dtype=torch.float64)
    if not torch.all((labels == 0.0) | (labels == 1.0)):
        raise ValueError("labels must be binary (0/1).")
    return labels, scores
