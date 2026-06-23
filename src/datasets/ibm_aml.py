"""
ibm_aml.py
==========
Loader do IBM Transactions for Anti-Money Laundering (AML) -> GraphSample.

Estrutura nativa (Kaggle: ealtman2019/ibm-transactions-for-anti-money-laundering-aml):
  colunas do *_Trans.csv:
    Timestamp, From Bank, Account, To Bank, Account.1,
    Amount Received, Receiving Currency, Amount Paid, Payment Currency,
    Payment Format, Is Laundering

Aqui a ARESTA é a transação e o NÓ é a CONTA — exatamente o alvo do produto
GraphGuard ("essa conta faz parte de uma rede suspeita?"). Como o dataset não
traz features de conta, nós as ENGENHEIRAMOS (as 11 features de rede, em
account_features.py). O rótulo `Is Laundering` é por transação; promovemos para
o nó: uma conta é fraude (1) se participou (origem OU destino) de qualquer
transação de lavagem.

Identidade da conta = (Banco, Account), pois o mesmo id de conta pode se repetir
entre bancos.

Nota sobre moedas: as features de volume somam `Amount Paid` ignorando a moeda
(as variantes Small são majoritariamente mono-moeda e o modelo é estrutural).
Para produção, normalize por câmbio ou use a moeda como feature de aresta.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .account_features import (
    FEATURE_NAMES,
    build_account_index,
    build_edge_index,
    extract_account_features,
)
from .common import GraphSample, random_split_mask, standardize


def _to_transactions(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Normaliza o CSV bruto para o contrato src/dst/amount/timestamp + flag de lavagem."""
    src = df["From Bank"].astype(str) + "_" + df["Account"].astype(str)
    dst = df["To Bank"].astype(str) + "_" + df["Account.1"].astype(str)

    amount = pd.to_numeric(df["Amount Paid"], errors="coerce").fillna(0.0)

    ts = pd.to_datetime(df["Timestamp"], errors="coerce")
    # timestamp em horas desde o início da janela (para o avg_forward_delay)
    t0 = ts.min()
    timestamp = (ts - t0).dt.total_seconds() / 3600.0
    timestamp = timestamp.fillna(0.0)

    is_laundering = pd.to_numeric(df["Is Laundering"], errors="coerce").fillna(0).astype(int)

    tx = pd.DataFrame(
        {"src": src, "dst": dst, "amount": amount.astype(np.float32),
         "timestamp": timestamp.astype(np.float32)}
    )
    return tx, is_laundering


def load_ibm_aml(
    root: str,
    trans_file: str,
    val_fraction: float = 0.15,
    compute_betweenness: bool = False,
    seed: int = 42,
    name: str | None = None,
) -> GraphSample:
    path = os.path.join(root, trans_file)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Arquivo do IBM AML não encontrado: {path}\n"
            "Baixe e extraia em data/raw/ibm-aml (ver docs/SEGUNDA-ETAPA.md)."
        )
    name = name or os.path.splitext(trans_file)[0]

    raw = pd.read_csv(path)
    tx, is_laundering = _to_transactions(raw)

    # --- índice de contas + arestas -----------------------------------------
    account_index = build_account_index(tx)
    edge_index = build_edge_index(tx, account_index)

    # --- rótulo por conta: participou de QUALQUER transação de lavagem -------
    n = len(account_index)
    y = np.zeros(n, dtype=np.int64)
    laundering_tx = tx[is_laundering.to_numpy() == 1]
    fraud_accounts = pd.unique(
        pd.concat([laundering_tx["src"], laundering_tx["dst"]], ignore_index=True)
    )
    for acc in fraud_accounts:
        y[account_index[acc]] = 1

    # --- features de rede engenheiradas (vetorizado) ------------------------
    x = extract_account_features(
        tx, account_index, compute_betweenness=compute_betweenness
    )
    x = standardize(x)

    # --- máscaras: todas as contas têm rótulo (0/1) -------------------------
    labeled_mask = np.ones(n, dtype=bool)
    train_mask, eval_mask = random_split_mask(labeled_mask, val_fraction, seed=seed)

    return GraphSample(
        name=name,
        x=x,
        edge_index=edge_index,
        y=y,
        labeled_mask=labeled_mask,
        train_mask=train_mask,
        eval_mask=eval_mask,
    )


__all__ = ["load_ibm_aml", "FEATURE_NAMES"]