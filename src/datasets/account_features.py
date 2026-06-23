"""
account_features.py
===================
Engenharia das features de rede por CONTA, de forma vetorizada.

O `graph_builder.py` original calcula as mesmas features iterando linha a linha
com networkx — perfeito para a demo sintética (centenas de transações), inviável
para o IBM AML (milhões de transações). Aqui as mesmas 11 features são
recalculadas com `groupby` do pandas, escalando para o HI-Small/LI-Small.

As features descrevem o comportamento *de vizinhança* de cada conta — é a tese
do GraphGuard: uma conta mula nova passa por qualquer filtro transacional, mas a
estrutura da rede dela a denuncia.

Contrato de entrada: DataFrame de transações com as colunas
    src, dst, amount, timestamp
(o mesmo contrato do data_generator sintético).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Mesma ordem canônica do graph_builder.py — mantém a tese consistente.
FEATURE_NAMES = [
    "in_degree",
    "out_degree",
    "total_in",
    "total_out",
    "flow_ratio",
    "n_senders",
    "n_receivers",
    "fanout",
    "avg_forward_delay",
    "betweenness",
    "amount_in_max",
]


def build_account_index(transactions: pd.DataFrame) -> dict[str, int]:
    """Mapeia cada conta (src ∪ dst) para um índice inteiro de nó, ordenado."""
    accounts = pd.Index(
        pd.unique(pd.concat([transactions["src"], transactions["dst"]], ignore_index=True))
    ).sort_values()
    return {acc: i for i, acc in enumerate(accounts)}


def extract_account_features(
    transactions: pd.DataFrame,
    account_index: dict[str, int],
    compute_betweenness: bool = False,
    betweenness_k: int = 500,
) -> np.ndarray:
    """
    Retorna uma matriz [n_contas, 11] na ordem de `account_index`.

    Tudo é vetorizado com groupby. `betweenness` é caro (BFS por amostra sobre o
    grafo inteiro) e por padrão fica desligado em grafos grandes — a coluna vira
    zero. Ligue `compute_betweenness=True` apenas em grafos pequenos.
    """
    n = len(account_index)
    idx = account_index

    # --- agregações por conta de DESTINO (entradas) -------------------------
    grp_in = transactions.groupby("dst")
    total_in = grp_in["amount"].sum()
    in_degree = grp_in["src"].nunique()          # contrapartes únicas de entrada
    first_in_ts = grp_in["timestamp"].min()       # 1º recebimento (p/ delay)
    # maior peso de aresta de entrada = maior total recebido de UMA contraparte
    # (mesma definição do graph_builder.py: agrega o multi-edge antes do max)
    edge_in = transactions.groupby(["dst", "src"])["amount"].sum()
    amount_in_max = edge_in.groupby(level="dst").max()

    # --- agregações por conta de ORIGEM (saídas) ----------------------------
    grp_out = transactions.groupby("src")
    total_out = grp_out["amount"].sum()
    out_degree = grp_out["dst"].nunique()         # contrapartes únicas de saída

    # --- atraso médio entre receber e repassar (velocidade da mula) ---------
    # Para cada transação de saída, delay = max(0, timestamp_saida - 1o_recebimento_do_src)
    out_tx = transactions[["src", "timestamp"]].copy()
    out_tx["first_in"] = out_tx["src"].map(first_in_ts)
    out_tx = out_tx.dropna(subset=["first_in"])             # só contas que receberam
    out_tx = out_tx[out_tx["timestamp"] >= out_tx["first_in"]]  # repasses após receber
    out_tx["delay"] = out_tx["timestamp"] - out_tx["first_in"]
    avg_forward_delay = out_tx.groupby("src")["delay"].mean()

    # --- monta a matriz na ordem dos nós ------------------------------------
    accounts = pd.Index(list(idx.keys()))

    def col(series: pd.Series, fill: float = 0.0) -> np.ndarray:
        return series.reindex(accounts).fillna(fill).to_numpy(dtype=np.float32)

    f_in_degree = col(in_degree)
    f_out_degree = col(out_degree)
    f_total_in = col(total_in)
    f_total_out = col(total_out)
    f_n_senders = f_in_degree            # após achatar multi-arestas, coincidem
    f_n_receivers = f_out_degree
    f_amount_in_max = col(amount_in_max)
    f_avg_delay = col(avg_forward_delay)

    f_flow_ratio = np.where(f_total_in > 0, f_total_out / np.maximum(f_total_in, 1e-9), 0.0)
    f_fanout = f_n_receivers / (f_n_senders + 1.0)

    f_betweenness = np.zeros(n, dtype=np.float32)
    if compute_betweenness:
        f_betweenness = _approx_betweenness(transactions, idx, k=betweenness_k)

    feats = np.column_stack(
        [
            f_in_degree,
            f_out_degree,
            f_total_in,
            f_total_out,
            f_flow_ratio.astype(np.float32),
            f_n_senders,
            f_n_receivers,
            f_fanout.astype(np.float32),
            f_avg_delay,
            f_betweenness,
            f_amount_in_max,
        ]
    ).astype(np.float32)
    return feats


def _approx_betweenness(
    transactions: pd.DataFrame, account_index: dict[str, int], k: int
) -> np.ndarray:
    """Betweenness aproximada via networkx (k fontes amostradas). Só p/ grafos pequenos."""
    import networkx as nx

    g = nx.DiGraph()
    agg = transactions.groupby(["src", "dst"])["amount"].sum().reset_index()
    g.add_weighted_edges_from(zip(agg["src"], agg["dst"], agg["amount"]))

    k = min(k, g.number_of_nodes())
    bc = nx.betweenness_centrality(g, k=k, seed=7) if g.number_of_nodes() > 2 else {}

    out = np.zeros(len(account_index), dtype=np.float32)
    for acc, i in account_index.items():
        out[i] = bc.get(acc, 0.0)
    return out


def build_edge_index(
    transactions: pd.DataFrame, account_index: dict[str, int]
) -> np.ndarray:
    """Constrói edge_index [2, n_edges] (arestas únicas src->dst) na indexação dos nós."""
    edges = transactions[["src", "dst"]].drop_duplicates()
    src = edges["src"].map(account_index).to_numpy(dtype=np.int64)
    dst = edges["dst"].map(account_index).to_numpy(dtype=np.int64)
    return np.vstack([src, dst])