"""
visualize.py
============
Visualização do grafo com nós coloridos pelo score de risco.

  - render_matplotlib : PNG estático (sempre funciona, bom para slides).
  - render_pyvis      : HTML interativo (arrastável, ótimo para a demo ao vivo).

A cor vai de verde (legítimo) a vermelho (alto risco), e o tamanho do nó
cresce com o score. Os clusters de fraude saltam aos olhos.
"""

from __future__ import annotations

import os

import networkx as nx
import pandas as pd


def _ensure_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)


def _color_for(score: float) -> str:
    """Verde -> amarelo -> vermelho conforme o score de risco."""
    if score < 0.33:
        return "#22c55e"
    if score < 0.66:
        return "#f59e0b"
    return "#ef4444"


def render_matplotlib(
    transactions: pd.DataFrame,
    scores: pd.Series,
    labels: dict[str, int] | None = None,
    path: str = "data/graph.png",
) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g = nx.DiGraph()
    for _, r in transactions.iterrows():
        g.add_edge(r["src"], r["dst"])

    pos = nx.spring_layout(g, seed=7, k=0.35, iterations=50)
    node_colors = [_color_for(scores.get(n, 0.0)) for n in g.nodes()]
    node_sizes = [60 + 340 * scores.get(n, 0.0) for n in g.nodes()]

    _ensure_dir(path)
    fig, ax = plt.subplots(figsize=(13, 9))
    nx.draw_networkx_edges(g, pos, alpha=0.15, width=0.6, ax=ax, arrows=False)
    nx.draw_networkx_nodes(
        g, pos, node_color=node_colors, node_size=node_sizes,
        linewidths=0.4, edgecolors="#33333355", ax=ax,
    )
    ax.set_title(
        "GraphGuard Pix — score de risco por conta\n"
        "verde = legítimo · amarelo = suspeito · vermelho = alta probabilidade de fraude",
        fontsize=13,
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def render_pyvis(
    transactions: pd.DataFrame,
    scores: pd.Series,
    labels: dict[str, int] | None = None,
    path: str = "data/graph.html",
) -> str:
    from pyvis.network import Network

    net = Network(
        height="720px", width="100%", directed=True,
        bgcolor="#ffffff", font_color="#333333",
    )
    net.barnes_hut(gravity=-8000, spring_length=120, spring_strength=0.02)

    g = nx.DiGraph()
    for _, r in transactions.iterrows():
        if g.has_edge(r["src"], r["dst"]):
            g[r["src"]][r["dst"]]["value"] += r["amount"]
        else:
            g.add_edge(r["src"], r["dst"], value=r["amount"])

    for node in g.nodes():
        score = float(scores.get(node, 0.0))
        true_label = labels.get(node) if labels else None
        title = f"{node}\nscore de risco: {score:.2f}"
        if true_label is not None:
            title += f"\nrótulo real: {'FRAUDE' if true_label else 'legítima'}"
        net.add_node(
            node,
            label="",
            title=title,
            color=_color_for(score),
            size=8 + 22 * score,
        )

    for u, v, d in g.edges(data=True):
        net.add_edge(u, v, color="#cccccc", width=0.5)

    _ensure_dir(path)
    net.save_graph(path)
    return path


if __name__ == "__main__":
    from data_generator import generate
    from graph_builder import extract_features
    from model import train_plan_b

    ds = generate()
    feats = extract_features(ds.transactions)
    res = train_plan_b(feats, ds.labels)

    png = render_matplotlib(ds.transactions, res.scores, ds.labels)
    html = render_pyvis(ds.transactions, res.scores, ds.labels)
    print(f"PNG salvo  : {png}")
    print(f"HTML salvo : {html}")
