from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class RGCNLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_relations: int,
        use_bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_relations = num_relations
        self.rel_weight = nn.Parameter(torch.empty(num_relations, in_dim, out_dim))
        self.self_loop_weight = nn.Parameter(torch.empty(in_dim, out_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim)) if use_bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.rel_weight)
        nn.init.xavier_uniform_(self.self_loop_weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.LongTensor,
        edge_type: torch.LongTensor,
    ) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError(f"Expected x with shape [N, D], got {tuple(x.shape)}")
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError(
                "Expected edge_index with shape [2, E], got "
                f"{tuple(edge_index.shape)}"
            )

        num_nodes = x.size(0)
        out = x @ self.self_loop_weight

        src_all = edge_index[0]
        dst_all = edge_index[1]

        for rel_id in range(self.num_relations):
            rel_mask = edge_type == rel_id
            if not torch.any(rel_mask):
                continue

            src = src_all[rel_mask]
            dst = dst_all[rel_mask]
            messages = x[src] @ self.rel_weight[rel_id]

            degree = torch.bincount(dst, minlength=num_nodes).clamp_min(1).to(messages.dtype)
            messages = messages / degree[dst].unsqueeze(-1)
            out.index_add_(0, dst, messages)

        if self.bias is not None:
            out = out + self.bias
        return out


@dataclass(frozen=True)
class ModelConfig:
    num_nodes: int
    num_relations: int
    hidden_dim: int = 128
    num_layers: int = 2
    pair_hidden_dim: int = 128
    dropout: float = 0.1


class BaseRGCNPairModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.config = config
        self.node_embedding = nn.Embedding(config.num_nodes, config.hidden_dim)
        self.layers = nn.ModuleList(
            [
                RGCNLayer(
                    in_dim=config.hidden_dim,
                    out_dim=config.hidden_dim,
                    num_relations=config.num_relations,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.dropout = config.dropout
        self.pair_head = nn.Sequential(
            nn.Linear(config.hidden_dim * 3, config.pair_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.pair_hidden_dim, 1),
        )
        self.hyperedge_head = nn.Sequential(
            nn.Linear(config.hidden_dim * 4, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        self.proj_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.node_embedding.weight)
        for layer in self.layers:
            layer.reset_parameters()
        for module in self.pair_head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        for module in self.hyperedge_head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        for module in self.proj_head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def encode(
        self,
        edge_index: torch.LongTensor,
        edge_type: torch.LongTensor,
    ) -> torch.Tensor:
        x = self.node_embedding.weight
        for layer_i, layer in enumerate(self.layers):
            x = layer(x=x, edge_index=edge_index, edge_type=edge_type)
            if layer_i < len(self.layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def score_pairs(
        self,
        node_embeddings: torch.Tensor,
        drug_index: torch.LongTensor,
        disease_index: torch.LongTensor,
    ) -> torch.Tensor:
        drug_h = node_embeddings[drug_index]
        disease_h = node_embeddings[disease_index]
        pair_feature = torch.cat((drug_h, disease_h, drug_h * disease_h), dim=-1)
        logits = self.pair_head(pair_feature).squeeze(-1)
        return logits

    def forward(
        self,
        edge_index: torch.LongTensor,
        edge_type: torch.LongTensor,
        drug_index: torch.LongTensor,
        disease_index: torch.LongTensor,
        ho_batch: Optional[Mapping[str, torch.Tensor]] = None,
        return_node_embeddings: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
        node_embeddings = self.encode(edge_index=edge_index, edge_type=edge_type)
        logits = self.score_pairs(
            node_embeddings=node_embeddings,
            drug_index=drug_index,
            disease_index=disease_index,
        )
        if ho_batch is not None:
            ho_drug_h = node_embeddings[ho_batch["drug_index"]]
            ho_protein_h = node_embeddings[ho_batch["protein_index"]]
            ho_pathway_h = node_embeddings[ho_batch["pathway_index"]]
            ho_disease_h = node_embeddings[ho_batch["disease_index"]]
            z_q = self.hyperedge_head(
                torch.cat((ho_drug_h, ho_protein_h, ho_pathway_h, ho_disease_h), dim=-1)
            )
            ho_outputs: dict[str, torch.Tensor] = {
                "z_q": z_q,
                "proj_drug": self.proj_head(ho_drug_h),
                "proj_protein": self.proj_head(ho_protein_h),
                "proj_pathway": self.proj_head(ho_pathway_h),
                "proj_disease": self.proj_head(ho_disease_h),
            }
            if "weight" in ho_batch:
                ho_outputs["w_q"] = ho_batch["weight"]
            if return_node_embeddings:
                ho_outputs["node_embeddings"] = node_embeddings
            return logits, ho_outputs
        if return_node_embeddings:
            return logits, node_embeddings
        return logits
