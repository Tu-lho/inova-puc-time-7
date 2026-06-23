"""
elliptic.py
===========
Loader do Elliptic Bitcoin Dataset -> GraphSample.

Estrutura nativa do dataset (Kaggle: ellipticco/elliptic-data-set):
  - elliptic_txs_features.csv : SEM cabeçalho. coluna 0 = txId, coluna 1 = time
        step (1..49), colunas 2..N = features (94 locais + 72 agregadas).
  - elliptic_txs_classes.csv  : cabeçalho txId,class. class ∈ {1=ilícito,
        2=lícito, unknown}.
  - elliptic_txs_edgelist.csv : cabeçalho txId1,txId2 (aresta direcionada).

Aqui o NÓ é a transação (semântica nativa do Elliptic). Usamos este dataset
como PRÉ-TREINO do backbone — a ressalva de que o produto final é
conta-como-nó está em docs/SEGUNDA-ETAPA.md. As 166 features anônimas vão
direto para o encoder específico do Elliptic.

Split temporal (Weber et al., 2019): time steps 1..34 = treino, 35..49 = teste.
Os nós "unknown" não entram na loss — só propagam mensagem no grafo.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .common import GraphSample, standardize


def load_elliptic(
    root: str,
    features_file: str = "elliptic_txs_features.csv",
    classes_file: str = "elliptic_txs_classes.csv",
    edgelist_file: str = "elliptic_txs_edgelist.csv",
    train_time_steps: tuple[int, int] = (1, 34),
    test_time_steps: tuple[int, int] = (35, 49),
    drop_unknown: bool = True,
    name: str = "elliptic",
) -> GraphSample:
    feat_path = os.path.join(root, features_file)
    cls_path = os.path.join(root, classes_file)
    edge_path = os.path.join(root, edgelist_file)
    for p in (feat_path, cls_path, edge_path):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Arquivo do Elliptic não encontrado: {p}\n"
                "Baixe e extraia em data/raw/elliptic (ver docs/SEGUNDA-ETAPA.md)."
            )

    # --- features (sem cabeçalho) -------------------------------------------
    feats = pd.read_csv(feat_path, header=None)
    tx_id = feats.iloc[:, 0].to_numpy()
    time_step = feats.iloc[:, 1].to_numpy().astype(np.int64)
    x = feats.iloc[:, 2:].to_numpy(dtype=np.float32)   # 166 features anônimas

    id_to_node = {t: i for i, t in enumerate(tx_id)}
    n = len(tx_id)

    # --- classes -> rótulos 0/1 + máscara de rotulados ----------------------
    classes = pd.read_csv(cls_path)
    cls_map = dict(zip(classes["txId"], classes["class"].astype(str)))
    y = np.zeros(n, dtype=np.int64)
    labeled_mask = np.zeros(n, dtype=bool)
    for t, node in id_to_node.items():
        c = cls_map.get(t, "unknown")
        if c == "1":               # ilícito
            y[node] = 1
            labeled_mask[node] = True
        elif c == "2":             # lícito
            y[node] = 0
            labeled_mask[node] = True
        # "unknown" -> permanece não rotulado (só mensagem)

    if not drop_unknown:
        labeled_mask[:] = True

    # --- arestas ------------------------------------------------------------
    edges = pd.read_csv(edge_path)
    src = edges["txId1"].map(id_to_node)
    dst = edges["txId2"].map(id_to_node)
    valid = src.notna() & dst.notna()
    edge_index = np.vstack(
        [src[valid].to_numpy(dtype=np.int64), dst[valid].to_numpy(dtype=np.int64)]
    )

    # --- split temporal -----------------------------------------------------
    tr_lo, tr_hi = train_time_steps
    te_lo, te_hi = test_time_steps
    in_train = (time_step >= tr_lo) & (time_step <= tr_hi)
    in_test = (time_step >= te_lo) & (time_step <= te_hi)
    train_mask = labeled_mask & in_train
    eval_mask = labeled_mask & in_test

    x = standardize(x)

    return GraphSample(
        name=name,
        x=x,
        edge_index=edge_index,
        y=y,
        labeled_mask=labeled_mask,
        train_mask=train_mask,
        eval_mask=eval_mask,
    )