from __future__ import annotations

from typing import Mapping

import torch
import torch.nn.functional as F


def ho_component_infonce_loss(
    ho_outputs: Mapping[str, torch.Tensor],
    tau: float = 0.1,
) -> torch.Tensor:
    if tau <= 0:
        raise ValueError("tau must be > 0 for InfoNCE.")

    z_q = ho_outputs["z_q"]
    proj_drug = ho_outputs["proj_drug"]
    proj_protein = ho_outputs["proj_protein"]
    proj_pathway = ho_outputs["proj_pathway"]
    proj_disease = ho_outputs["proj_disease"]
    w_q = ho_outputs.get("w_q")

    if z_q.dim() != 2:
        raise ValueError(f"Expected z_q to have shape [B, D], got {tuple(z_q.shape)}")
    batch_size = z_q.size(0)
    if batch_size == 0:
        raise ValueError("HO batch is empty.")

    query = F.normalize(z_q, p=2, dim=-1)
    targets = torch.arange(batch_size, device=z_q.device)

    component_losses: list[torch.Tensor] = []
    for component_proj in (proj_drug, proj_protein, proj_pathway, proj_disease):
        if component_proj.shape != z_q.shape:
            raise ValueError(
                "Projection shape mismatch in HO outputs. "
                f"Expected {tuple(z_q.shape)}, got {tuple(component_proj.shape)}"
            )
        keys = F.normalize(component_proj, p=2, dim=-1)
        logits = (query @ keys.transpose(0, 1)) / tau
        component_losses.append(F.cross_entropy(logits, targets, reduction="none"))

    per_quad_loss = torch.stack(component_losses, dim=1).mean(dim=1)
    if w_q is None:
        return per_quad_loss.mean()

    if w_q.shape != (batch_size,):
        raise ValueError(
            f"Expected w_q to have shape ({batch_size},), got {tuple(w_q.shape)}"
        )
    weights = w_q.to(per_quad_loss.dtype)
    normalizer = weights.sum().clamp_min(torch.finfo(per_quad_loss.dtype).eps)
    return (per_quad_loss * weights).sum() / normalizer
