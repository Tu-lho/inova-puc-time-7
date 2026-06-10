"""
graph_builder.py
================
Constrói o grafo direcionado de transações e extrai features por nó.

Estas features são o coração da tese do projeto: elas descrevem o
comportamento *de rede* de cada conta — não a transação isolada. É por isso
que uma conta mula nova, que passaria por qualquer filtro transacional,
acende aqui: a estrutura da vizinhança dela a denuncia.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

# ordem canônica das features (usada pelo modelo e pela visualização)
FEATURE_NAMES = [
    "in_degree",
    "out_degree",
    "total_in",
    "total_out",
    "flow_ratio",        # total_out / total_in  (mulas ~ 1: tudo que entra, sai)
    "n_senders",
    "n_receivers",
    "fanout",            # n_receivers / (n_senders + 1)
    "avg_forward_delay", # tempo médio entre receber e repassar (velocidade)
    "betweenness",       # centralidade de intermediação (mulas são pontes)
    "amount_in_max",
]


def build_graph(transactions: pd.DataFrame) -> nx.DiGraph:
    """Cria um MultiDiGraph achatado em DiGraph com atributos agregados."""
    g = nx.DiGraph()
    for _, row in transactions.iterrows():
        src, dst = row["src"], row["dst"]
        if g.has_edge(src, dst):
            g[src][dst]["weight"] += row["amount"]
            g[src][dst]["count"] += 1
        else:
            g.add_edge(src, dst, weight=row["amount"], count=1)
    return g


def extract_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """Retorna um DataFrame indexado por conta com as features de rede."""
    accounts = sorted(set(transactions["src"]) | set(transactions["dst"]))
    g = build_graph(transactions)

    # centralidade de intermediação aproximada (k amostras p/ velocidade)
    k = min(len(accounts), 100)
    betweenness = nx.betweenness_centrality(g, k=k, seed=7) if len(g) > 2 else {}

    # pré-computa, por conta, os tempos de recebimento e de envio
    received_ts: dict[str, list[float]] = {a: [] for a in accounts}
    sent_ts: dict[str, list[float]] = {a: [] for a in accounts}
    for _, row in transactions.iterrows():
        received_ts[row["dst"]].append(row["timestamp"])
        sent_ts[row["src"]].append(row["timestamp"])

    feats: list[dict] = []
    for acc in accounts:
        in_edges = list(g.in_edges(acc, data=True))
        out_edges = list(g.out_edges(acc, data=True))

        total_in = sum(d["weight"] for _, _, d in in_edges)
        total_out = sum(d["weight"] for _, _, d in out_edges)
        in_degree = len(in_edges)
        out_degree = len(out_edges)
        n_senders = len({u for u, _, _ in in_edges})
        n_receivers = len({v for _, v, _ in out_edges})

        amount_in_max = max((d["weight"] for _, _, d in in_edges), default=0.0)

        # atraso médio entre o primeiro recebimento e os repasses subsequentes
        rec = sorted(received_ts[acc])
        snt = sorted(sent_ts[acc])
        if rec and snt:
            first_in = rec[0]
            delays = [s - first_in for s in snt if s >= first_in]
            avg_forward_delay = float(np.mean(delays)) if delays else 0.0
        else:
            avg_forward_delay = 0.0

        feats.append(
            dict(
                account=acc,
                in_degree=in_degree,
                out_degree=out_degree,
                total_in=round(total_in, 2),
                total_out=round(total_out, 2),
                flow_ratio=round(total_out / total_in, 3) if total_in > 0 else 0.0,
                n_senders=n_senders,
                n_receivers=n_receivers,
                fanout=round(n_receivers / (n_senders + 1), 3),
                avg_forward_delay=round(avg_forward_delay, 3),
                betweenness=round(betweenness.get(acc, 0.0), 5),
                amount_in_max=round(amount_in_max, 2),
            )
        )

    df = pd.DataFrame(feats).set_index("account")
    return df[FEATURE_NAMES]


if __name__ == "__main__":
    from data_generator import generate

    ds = generate()
    g = build_graph(ds.transactions)
    print(f"Nós (contas) : {g.number_of_nodes()}")
    print(f"Arestas      : {g.number_of_edges()}")

    feats = extract_features(ds.transactions)
    print("\nFeatures (amostra de contas de fraude vs legítimas):")
    feats = feats.assign(label=[ds.labels[a] for a in feats.index])
    print("\n--- Contas em fraude ---")
    print(feats[feats.label == 1].head(4).to_string())
    print("\n--- Contas legítimas ---")
    print(feats[feats.label == 0].head(4).to_string())
