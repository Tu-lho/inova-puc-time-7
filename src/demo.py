"""
demo.py
=======
Runner ponta-a-ponta do GraphGuard Pix. Este é o arquivo da demo ao vivo.

Fluxo:
  1. gera transações sintéticas com fraude injetada
  2. constrói o grafo e extrai features de rede
  3. treina o modelo (Plano B por padrão; Plano A se disponível e pedido)
  4. avalia e imprime a métrica-chave do pitch
  5. gera a visualização (PNG + HTML interativo)

Uso:
    python src/demo.py                # Plano B (sempre funciona)
    python src/demo.py --plan a       # tenta GraphSAGE (precisa torch-geometric)
    python src/demo.py --seed 7       # outra amostra
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_generator import GeneratorConfig, generate  # noqa: E402
from graph_builder import FEATURE_NAMES, extract_features  # noqa: E402
from model import (  # noqa: E402
    feature_importance,
    plan_a_available,
    train_plan_a,
    train_plan_b,
)
from visualize import render_matplotlib, render_pyvis  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def transaction_filter_baseline(features, labels, amount_threshold_pct=0.9):
    """
    Baseline ingênuo que imita um filtro por transação: marca como suspeita
    apenas a conta que recebeu um valor muito acima do normal (alto
    amount_in_max). É o tipo de regra que pega o hub mas NÃO pega as mulas
    nem os sinks — que é justamente o que a GNN/feature de rede captura.
    """
    threshold = features["amount_in_max"].quantile(amount_threshold_pct)
    flagged = set(features.index[features["amount_in_max"] >= threshold])
    fraud = {a for a, y in labels.items() if y == 1}
    caught = flagged & fraud
    recall = len(caught) / max(len(fraud), 1)
    return recall, flagged


def banner(text: str) -> None:
    print("\n" + "=" * 64)
    print(f"  {text}")
    print("=" * 64)


def main() -> None:
    parser = argparse.ArgumentParser(description="GraphGuard Pix — demo")
    parser.add_argument("--plan", choices=["a", "b"], default="b")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    banner("1. Gerando transações Pix sintéticas")
    ds = generate(GeneratorConfig(seed=args.seed))
    n_fraud = ds.n_fraud_accounts
    print(f"  Transações        : {len(ds.transactions)}")
    print(f"  Contas            : {ds.n_accounts}")
    print(f"  Contas em fraude  : {n_fraud} ({n_fraud / ds.n_accounts:.1%})")

    banner("2. Construindo grafo e extraindo features de rede")
    feats = extract_features(ds.transactions)
    print(f"  Features por conta: {len(FEATURE_NAMES)}")
    print(f"  {', '.join(FEATURE_NAMES)}")

    banner("3. Treinando o modelo de detecção")
    use_plan_a = args.plan == "a" and plan_a_available()
    if args.plan == "a" and not plan_a_available():
        print("  torch-geometric não encontrado — caindo para o Plano B.")
    if use_plan_a:
        print("  Plano A: GraphSAGE (2 camadas)")
        res = train_plan_a(feats, ds.transactions, ds.labels, seed=args.seed)
    else:
        print("  Plano B: features de rede + GradientBoosting")
        res = train_plan_b(feats, ds.labels, seed=args.seed)

    banner("4. Avaliação")
    print(f"  Plano usado       : {res.plan}")
    print(f"  AUC-ROC           : {res.auc:.3f}")
    print(f"  Average Precision : {res.avg_precision:.3f}")

    if res.plan == "B":
        print("\n  Features mais importantes:")
        for name, imp in feature_importance(res, FEATURE_NAMES).head(5).items():
            print(f"    {name:<20} {imp:.3f}")

    # --- a métrica-chave do pitch ---
    baseline_recall, _ = transaction_filter_baseline(feats, ds.labels)
    gg_recall = _model_recall(res.scores, ds.labels, threshold=0.5)
    banner("5. O número que vende o produto")
    print(f"  Filtro por transação (baseline) pega : {baseline_recall:.0%} das contas mula")
    print(f"  GraphGuard (rede)               pega : {gg_recall:.0%} das contas mula")
    uplift = gg_recall - baseline_recall
    print(f"  Ganho de cobertura                   : +{uplift:.0%} pontos")
    print("\n  Tradução para o pitch:")
    print("  \"Um filtro por transação pega o hub que recebeu o valor grande,")
    print("   mas deixa passar as mulas e os sinks. O GraphGuard pega a rede inteira.\"")

    banner("6. Gerando visualização")
    png = render_matplotlib(
        ds.transactions, res.scores, ds.labels, path=os.path.join(ROOT, "data/graph.png")
    )
    html = render_pyvis(
        ds.transactions, res.scores, ds.labels, path=os.path.join(ROOT, "data/graph.html")
    )
    print(f"  PNG  : {png}")
    print(f"  HTML : {html}  (abra no navegador para a demo interativa)")
    print()


def _model_recall(scores, labels, threshold=0.5):
    fraud = {a for a, y in labels.items() if y == 1}
    flagged = {a for a in scores.index if scores[a] >= threshold}
    caught = flagged & fraud
    return len(caught) / max(len(fraud), 1)


if __name__ == "__main__":
    main()
