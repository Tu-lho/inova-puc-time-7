"""
common.py
=========
Contrato comum que todo loader de dataset entrega ao treinador.

A ideia central do projeto: não importa se o dataset é o Elliptic
(transação-como-nó) ou o IBM AML (conta-como-nó), tudo é normalizado para a
MESMA estrutura `GraphSample`. O encoder por dataset (em src/gnn/model.py)
cuida de mapear as features nativas — que têm dimensões diferentes — para o
espaço comum onde o backbone GraphSAGE opera.

`GraphSample` carrega o suficiente para treinar e avaliar:
  - x          : features dos nós            [n_nodes, in_dim]
  - edge_index : arestas direcionadas        [2, n_edges]
  - y          : rótulos 0/1                  [n_nodes]
  - labeled_mask : nós COM rótulo confiável (no Elliptic ~21% são "unknown")
  - train_mask / eval_mask : partição treino/avaliação (já interseccionada
                             com labeled_mask)
  - name, in_dim : metadados
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GraphSample:
    name: str
    x: np.ndarray            # [n_nodes, in_dim] float32
    edge_index: np.ndarray   # [2, n_edges] int64
    y: np.ndarray            # [n_nodes] int64 (0/1)
    labeled_mask: np.ndarray  # [n_nodes] bool
    train_mask: np.ndarray   # [n_nodes] bool
    eval_mask: np.ndarray    # [n_nodes] bool

    @property
    def n_nodes(self) -> int:
        return int(self.x.shape[0])

    @property
    def in_dim(self) -> int:
        return int(self.x.shape[1])

    @property
    def n_edges(self) -> int:
        return int(self.edge_index.shape[1])

    @property
    def fraud_rate(self) -> float:
        lab = self.y[self.labeled_mask]
        return float(lab.mean()) if lab.size else 0.0

    def summary(self) -> str:
        return (
            f"{self.name}: nós={self.n_nodes:,} arestas={self.n_edges:,} "
            f"features={self.in_dim} rotulados={int(self.labeled_mask.sum()):,} "
            f"(treino={int(self.train_mask.sum()):,} aval={int(self.eval_mask.sum()):,}) "
            f"taxa_fraude={self.fraud_rate:.4%}"
        )

    def to_pyg(self):
        """Converte para um objeto torch_geometric.data.Data (import preguiçoso)."""
        import torch
        from torch_geometric.data import Data

        data = Data(
            x=torch.tensor(self.x, dtype=torch.float),
            edge_index=torch.tensor(self.edge_index, dtype=torch.long),
            y=torch.tensor(self.y, dtype=torch.long),
        )
        data.labeled_mask = torch.tensor(self.labeled_mask, dtype=torch.bool)
        data.train_mask = torch.tensor(self.train_mask, dtype=torch.bool)
        data.eval_mask = torch.tensor(self.eval_mask, dtype=torch.bool)
        return data


def standardize(x: np.ndarray) -> np.ndarray:
    """Padroniza as features (z-score por coluna). Crucial para GNN convergir."""
    x = x.astype(np.float32, copy=False)
    mu = x.mean(axis=0, keepdims=True)
    sigma = x.std(axis=0, keepdims=True)
    return (x - mu) / (sigma + 1e-6)


def random_split_mask(
    labeled_mask: np.ndarray, eval_fraction: float, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """
    Divide os nós ROTULADOS em treino/avaliação.

    eval_fraction == 0.0  -> tudo é treino   (ex.: HI-Small sem holdout)
    eval_fraction == 1.0  -> tudo é avaliação (ex.: LI-Small, estágio de validação)
    """
    n = labeled_mask.shape[0]
    train_mask = np.zeros(n, dtype=bool)
    eval_mask = np.zeros(n, dtype=bool)

    labeled_idx = np.flatnonzero(labeled_mask)
    if labeled_idx.size == 0:
        return train_mask, eval_mask

    if eval_fraction >= 1.0:
        eval_mask[labeled_idx] = True
        return train_mask, eval_mask
    if eval_fraction <= 0.0:
        train_mask[labeled_idx] = True
        return train_mask, eval_mask

    rng = np.random.default_rng(seed)
    shuffled = labeled_idx.copy()
    rng.shuffle(shuffled)
    n_eval = int(round(eval_fraction * shuffled.size))
    eval_mask[shuffled[:n_eval]] = True
    train_mask[shuffled[n_eval:]] = True
    return train_mask, eval_mask