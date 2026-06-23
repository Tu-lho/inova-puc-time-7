"""
model.py (gnn)
==============
A GNN do GraphGuard para a segunda etapa: backbone GraphSAGE COMPARTILHADO
entre datasets + um ENCODER de entrada específico por dataset.

Por que essa separação? Os datasets têm espaços de features diferentes
(Elliptic: 166 features dadas; IBM AML: 11 features de rede engenheiradas).
Pesos de message-passing não transferem se a dimensão de entrada muda. A
solução padrão da literatura:

    x  --[encoder específico do dataset]-->  h (dim comum)
       --[backbone GraphSAGE compartilhado]-->  h'
       --[cabeça de classificação compartilhada]-->  logits

O encoder é reinicializado a cada dataset (in_dim diferente). O backbone e a
cabeça são SALVOS e CARREGADOS entre estágios — é isso que materializa o
transfer learning: pré-treina no Elliptic, transfere o backbone, faz
fine-tuning no IBM AML.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class GraphGuardGNN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.dropout = dropout

        # --- encoder específico do dataset (NÃO transferido) ----------------
        self.encoder = nn.Linear(in_dim, hidden_dim)

        # --- backbone GraphSAGE compartilhado (transferido) -----------------
        self.convs = nn.ModuleList(
            [SAGEConv(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )

        # --- cabeça de classificação compartilhada (transferida) ------------
        self.head = nn.Linear(hidden_dim, 2)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.encoder(x))
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return self.head(h)

    # ------------------------------------------------------------------
    # Transferência: salva/carrega apenas backbone + cabeça (não o encoder)
    # ------------------------------------------------------------------
    def backbone_state_dict(self) -> dict:
        state = {f"convs.{k}": v for k, v in self.convs.state_dict().items()}
        state.update({f"head.{k}": v for k, v in self.head.state_dict().items()})
        return state

    def load_backbone(self, state: dict, strict: bool = True) -> None:
        convs_state = {
            k[len("convs."):]: v for k, v in state.items() if k.startswith("convs.")
        }
        head_state = {
            k[len("head."):]: v for k, v in state.items() if k.startswith("head.")
        }
        self.convs.load_state_dict(convs_state, strict=strict)
        self.head.load_state_dict(head_state, strict=strict)


def save_backbone(model: GraphGuardGNN, path: str) -> None:
    torch.save(model.backbone_state_dict(), path)


def load_backbone_into(model: GraphGuardGNN, path: str, strict: bool = True) -> None:
    state = torch.load(path, map_location="cpu")
    model.load_backbone(state, strict=strict)