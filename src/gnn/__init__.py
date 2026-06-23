"""Pacote da GNN do GraphGuard: modelo, métricas e treino por transfer learning."""

from __future__ import annotations

from .metrics import evaluate, recall_at_fpr
from .model import GraphGuardGNN
from .train import run_pipeline, train_stage

__all__ = [
    "GraphGuardGNN",
    "run_pipeline",
    "train_stage",
    "evaluate",
    "recall_at_fpr",
]