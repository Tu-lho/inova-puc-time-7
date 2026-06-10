"""
model.py
========
Modelo de detecção de fraude com dois planos, conforme o escopo do hackathon.

  Plano B (padrão, sempre roda):
      features de rede  ->  GradientBoosting (scikit-learn).
      Não é uma GNN, mas demonstra a mesma tese — features de rede detectam
      o que features de transação não detectam — e treina em segundos.

  Plano A (opcional, se torch-geometric estiver instalado):
      GraphSAGE de 2 camadas para classificação de nós.
      Importado de forma preguiçosa para que a ausência de PyG nunca quebre
      o pipeline.

Use `train_plan_b` para a demo garantida. `train_plan_a` é tentado apenas se
explicitamente chamado.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split


@dataclass
class Result:
    scores: pd.Series          # score de risco por conta (0..1)
    auc: float
    avg_precision: float
    model: object
    plan: str


# ----------------------------------------------------------------------
# Plano B — features de rede + classificador clássico
# ----------------------------------------------------------------------
def train_plan_b(
    features: pd.DataFrame,
    labels: dict[str, int],
    test_size: float = 0.3,
    seed: int = 42,
) -> Result:
    X = features.values
    y = np.array([labels[a] for a in features.index])

    X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
        X, y, features.index, test_size=test_size, random_state=seed, stratify=y
    )

    clf = GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.1, random_state=seed
    )
    clf.fit(X_tr, y_tr)

    proba_te = clf.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, proba_te)
    ap = average_precision_score(y_te, proba_te)

    # score para TODAS as contas (treino + teste) para a visualização
    all_scores = clf.predict_proba(X)[:, 1]
    scores = pd.Series(all_scores, index=features.index, name="risk_score")

    return Result(scores=scores, auc=auc, avg_precision=ap, model=clf, plan="B")


def feature_importance(result: Result, feature_names: list[str]) -> pd.Series:
    """Importância das features no modelo do Plano B (poderoso no pitch)."""
    if not hasattr(result.model, "feature_importances_"):
        return pd.Series(dtype=float)
    imp = pd.Series(result.model.feature_importances_, index=feature_names)
    return imp.sort_values(ascending=False)


# ----------------------------------------------------------------------
# Plano A — GraphSAGE (opcional). Só roda se torch-geometric existir.
# ----------------------------------------------------------------------
def plan_a_available() -> bool:
    try:
        import torch  # noqa: F401
        import torch_geometric  # noqa: F401

        return True
    except ImportError:
        return False


def train_plan_a(
    features: pd.DataFrame,
    transactions: pd.DataFrame,
    labels: dict[str, int],
    epochs: int = 60,
    seed: int = 42,
) -> Result:
    """GraphSAGE de 2 camadas. Requer torch + torch_geometric."""
    import torch
    import torch.nn.functional as F
    from torch_geometric.data import Data
    from torch_geometric.nn import SAGEConv

    torch.manual_seed(seed)
    accounts = list(features.index)
    idx = {a: i for i, a in enumerate(accounts)}

    x = torch.tensor(features.values, dtype=torch.float)
    # normaliza as features (importante para GNN)
    x = (x - x.mean(0)) / (x.std(0) + 1e-6)

    edges = [[idx[s], idx[d]] for s, d in zip(transactions["src"], transactions["dst"])]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    y = torch.tensor([labels[a] for a in accounts], dtype=torch.long)

    n = len(accounts)
    perm = torch.randperm(n)
    train_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[perm[: int(0.7 * n)]] = True
    test_mask = ~train_mask

    data = Data(x=x, edge_index=edge_index, y=y)

    class SAGE(torch.nn.Module):
        def __init__(self, in_dim, hidden=32):
            super().__init__()
            self.c1 = SAGEConv(in_dim, hidden)
            self.c2 = SAGEConv(hidden, 2)

        def forward(self, d):
            h = F.relu(self.c1(d.x, d.edge_index))
            h = F.dropout(h, p=0.3, training=self.training)
            return self.c2(h, d.edge_index)

    model = SAGE(x.size(1))
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    # peso de classe (fraude é minoria)
    weight = torch.tensor([1.0, (y == 0).sum() / max((y == 1).sum(), 1)])

    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(data)
        loss = F.cross_entropy(out[train_mask], y[train_mask], weight=weight)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        proba = F.softmax(model(data), dim=1)[:, 1].numpy()

    auc = roc_auc_score(y[test_mask].numpy(), proba[test_mask.numpy()])
    ap = average_precision_score(y[test_mask].numpy(), proba[test_mask.numpy()])
    scores = pd.Series(proba, index=accounts, name="risk_score")
    return Result(scores=scores, auc=auc, avg_precision=ap, model=model, plan="A")


if __name__ == "__main__":
    from data_generator import generate
    from graph_builder import FEATURE_NAMES, extract_features

    ds = generate()
    feats = extract_features(ds.transactions)

    res = train_plan_b(feats, ds.labels)
    print(f"[Plano B] AUC-ROC: {res.auc:.3f}  |  Avg Precision: {res.avg_precision:.3f}")
    print("\nFeatures mais importantes:")
    print(feature_importance(res, FEATURE_NAMES).head(6).to_string())

    print(f"\nPlano A (GraphSAGE) disponível: {plan_a_available()}")
