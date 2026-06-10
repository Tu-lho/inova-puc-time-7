"""
data_generator.py
==================
Gera transações Pix sintéticas com redes de fraude injetadas.

Cria uma população de contas legítimas que transacionam de forma
descentralizada e, sobre ela, injeta dois padrões de fraude com rótulos
conhecidos:

  - smurfing : um hub recebe um valor grande de uma vítima e o distribui
               em vários valores menores para contas mula.
  - fan-out  : o dinheiro passa por múltiplas camadas de contas para
               diluir o rastro (layering).

A saída é um DataFrame de transações e um dicionário de rótulos por conta
(0 = legítima, 1 = envolvida em fraude). Os rótulos existem apenas para
treino e avaliação — em produção eles seriam parcialmente conhecidos
(contas já denunciadas) e o modelo generalizaria para o resto.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class GeneratorConfig:
    n_legit_accounts: int = 220
    n_legit_transactions: int = 700
    n_legit_merchants: int = 12      # confundidores: alto grau de entrada + repasse rápido
    n_smurfing_rings: int = 4
    n_fanout_rings: int = 3
    seed: int = 42
    # janela temporal simulada (em horas)
    horizon_hours: int = 24 * 30


@dataclass
class Dataset:
    transactions: pd.DataFrame
    labels: dict[str, int] = field(default_factory=dict)

    @property
    def n_accounts(self) -> int:
        accounts = set(self.transactions["src"]) | set(self.transactions["dst"])
        return len(accounts)

    @property
    def n_fraud_accounts(self) -> int:
        return sum(self.labels.values())


def _acc_id(prefix: str, i: int) -> str:
    """ID pseudonimizado da conta (em produção seria um hash, não o CPF)."""
    return f"{prefix}_{i:05d}"


def generate(config: GeneratorConfig | None = None) -> Dataset:
    cfg = config or GeneratorConfig()
    rng = random.Random(cfg.seed)
    np_rng = np.random.default_rng(cfg.seed)

    rows: list[dict] = []
    labels: dict[str, int] = {}

    legit_accounts = [_acc_id("acc", i) for i in range(cfg.n_legit_accounts)]
    for acc in legit_accounts:
        labels[acc] = 0

    def ts() -> float:
        return float(np_rng.uniform(0, cfg.horizon_hours))

    # ------------------------------------------------------------------
    # 1. Tráfego legítimo: transferências descentralizadas e irregulares.
    # ------------------------------------------------------------------
    for _ in range(cfg.n_legit_transactions):
        src, dst = rng.sample(legit_accounts, 2)
        amount = round(np_rng.lognormal(mean=4.2, sigma=1.0), 2)  # ~R$66 mediano
        rows.append(
            dict(src=src, dst=dst, amount=amount, timestamp=ts(), pattern="legit")
        )

    # ------------------------------------------------------------------
    # 1b. Comerciantes legítimos (CONFUNDIDORES): recebem de muitos clientes
    #     e repassam rápido a poucos fornecedores. Sem isso, qualquer conta
    #     com repasse rápido pareceria fraude — o que tornaria o problema
    #     trivial e a demo pouco convincente.
    # ------------------------------------------------------------------
    merchants = [_acc_id("merchant", i) for i in range(cfg.n_legit_merchants)]
    for merch in merchants:
        labels[merch] = 0
        n_customers = rng.randint(8, 20)
        t_base = ts()
        received_total = 0.0
        for _ in range(n_customers):
            customer = rng.choice(legit_accounts)
            amt = round(np_rng.lognormal(mean=4.0, sigma=0.8), 2)
            received_total += amt
            rows.append(
                dict(
                    src=customer,
                    dst=merch,
                    amount=amt,
                    timestamp=t_base + np_rng.uniform(0, 48),
                    pattern="legit",
                )
            )
        # repassa a fornecedores com atraso curto-moderado (overlap com mulas)
        n_suppliers = rng.randint(2, 4)
        for _ in range(n_suppliers):
            supplier = rng.choice(legit_accounts)
            rows.append(
                dict(
                    src=merch,
                    dst=supplier,
                    amount=round(received_total / n_suppliers * np_rng.uniform(0.7, 0.95), 2),
                    timestamp=t_base + np_rng.uniform(0.5, 6.0),
                    pattern="legit",
                )
            )

    fraud_idx = 0

    # ------------------------------------------------------------------
    # 2. Smurfing: vítima -> hub -> várias mulas -> saída.
    # ------------------------------------------------------------------
    for r in range(cfg.n_smurfing_rings):
        victim = rng.choice(legit_accounts)  # a vítima é uma conta legítima
        hub = _acc_id("mule_smurf_hub", fraud_idx)
        labels[hub] = 1
        fraud_idx += 1

        big_amount = round(np_rng.uniform(3000, 9000), 2)
        t0 = ts()
        rows.append(
            dict(src=victim, dst=hub, amount=big_amount, timestamp=t0, pattern="smurf")
        )

        n_mules = rng.randint(4, 7)
        per = big_amount / n_mules
        for m in range(n_mules):
            mule = _acc_id("mule_smurf", fraud_idx)
            labels[mule] = 1
            fraud_idx += 1
            # o hub repassa quase imediatamente (minutos = fração de hora)
            t1 = t0 + np_rng.uniform(0.01, 0.5)
            amt = round(per * np_rng.uniform(0.85, 1.0), 2)
            rows.append(
                dict(src=hub, dst=mule, amount=amt, timestamp=t1, pattern="smurf")
            )
            # a mula saca / repassa para um destino de saída
            sink = _acc_id("sink", fraud_idx)
            labels[sink] = 1
            fraud_idx += 1
            t2 = t1 + np_rng.uniform(0.01, 0.8)
            rows.append(
                dict(
                    src=mule,
                    dst=sink,
                    amount=round(amt * np_rng.uniform(0.9, 0.99), 2),
                    timestamp=t2,
                    pattern="smurf",
                )
            )

    # ------------------------------------------------------------------
    # 3. Fan-out / layering: cadeia em camadas que dilui o rastro.
    # ------------------------------------------------------------------
    for r in range(cfg.n_fanout_rings):
        victim = rng.choice(legit_accounts)
        layer0 = _acc_id("mule_fan_l0", fraud_idx)
        labels[layer0] = 1
        fraud_idx += 1

        amount = round(np_rng.uniform(5000, 12000), 2)
        t0 = ts()
        rows.append(
            dict(src=victim, dst=layer0, amount=amount, timestamp=t0, pattern="fanout")
        )

        current_layer = [(layer0, amount, t0)]
        n_layers = rng.randint(2, 3)
        for depth in range(n_layers):
            next_layer = []
            for (node, amt, t) in current_layer:
                n_children = rng.randint(2, 3)
                split = amt / n_children
                for c in range(n_children):
                    child = _acc_id(f"mule_fan_l{depth + 1}", fraud_idx)
                    labels[child] = 1
                    fraud_idx += 1
                    t_next = t + np_rng.uniform(0.02, 1.0)
                    amt_c = round(split * np_rng.uniform(0.9, 0.99), 2)
                    rows.append(
                        dict(
                            src=node,
                            dst=child,
                            amount=amt_c,
                            timestamp=t_next,
                            pattern="fanout",
                        )
                    )
                    next_layer.append((child, amt_c, t_next))
            current_layer = next_layer

    df = pd.DataFrame(rows)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return Dataset(transactions=df, labels=labels)


if __name__ == "__main__":
    ds = generate()
    print(f"Transações geradas : {len(ds.transactions)}")
    print(f"Contas totais      : {ds.n_accounts}")
    print(f"Contas em fraude   : {ds.n_fraud_accounts}")
    print(f"Taxa de fraude     : {ds.n_fraud_accounts / ds.n_accounts:.1%}")
    print("\nAmostra de transações:")
    print(ds.transactions.head(8).to_string(index=False))
