"""
metrics.py
==========
Métricas de avaliação para detecção de fraude com classe desbalanceada.

A métrica que vende o produto não é acurácia (inútil com 0,05% de fraude) e nem
só AUROC. É o **recall com no máximo 5% de falsos positivos** — espelha a
restrição real: investigadores só conseguem revisar um número limitado de
transações flagradas. Esse é o benchmark prático do Elliptic.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def recall_at_fpr(y_true: np.ndarray, scores: np.ndarray, max_fpr: float = 0.05) -> float:
    """Maior recall (TPR) alcançável mantendo a taxa de falsos positivos <= max_fpr."""
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return float("nan")
    fpr, tpr, _ = roc_curve(y_true, scores)
    ok = fpr <= max_fpr
    return float(tpr[ok].max()) if ok.any() else 0.0


def evaluate(y_true: np.ndarray, scores: np.ndarray, max_fpr: float = 0.05) -> dict:
    """AUROC, Average Precision e recall@FPR — o pacote completo para o pitch."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    metrics = {
        "n": int(y_true.size),
        "n_fraud": int(y_true.sum()),
        "fraud_rate": float(y_true.mean()) if y_true.size else float("nan"),
    }
    if 0 < y_true.sum() < y_true.size:
        metrics["auroc"] = float(roc_auc_score(y_true, scores))
        metrics["ap"] = float(average_precision_score(y_true, scores))
        metrics[f"recall@fpr{max_fpr:g}"] = recall_at_fpr(y_true, scores, max_fpr)
    else:
        metrics["auroc"] = float("nan")
        metrics["ap"] = float("nan")
        metrics[f"recall@fpr{max_fpr:g}"] = float("nan")
    return metrics


def format_metrics(m: dict) -> str:
    parts = [f"n={m['n']:,}", f"fraude={m['n_fraud']:,} ({m['fraud_rate']:.4%})"]
    for k, v in m.items():
        if k in ("n", "n_fraud", "fraud_rate"):
            continue
        parts.append(f"{k}={v:.4f}")
    return "  ".join(parts)