"""
Pacote de loaders de datasets de fraude financeira.

Cada loader entrega um `GraphSample` (contrato comum em common.py). A função
`load_dataset` despacha pela chave `kind` da configuração.
"""

from __future__ import annotations

from .common import GraphSample
from .elliptic import load_elliptic
from .ibm_aml import load_ibm_aml


def load_dataset(cfg: dict, seed: int = 42) -> GraphSample:
    """
    Constrói um GraphSample a partir de um bloco de configuração de dataset.

    `cfg` é uma das entradas de `datasets:` no YAML (já com a chave `kind`).
    """
    kind = cfg["kind"]

    if kind == "elliptic":
        return load_elliptic(
            root=cfg["root"],
            features_file=cfg.get("features_file", "elliptic_txs_features.csv"),
            classes_file=cfg.get("classes_file", "elliptic_txs_classes.csv"),
            edgelist_file=cfg.get("edgelist_file", "elliptic_txs_edgelist.csv"),
            train_time_steps=tuple(cfg.get("train_time_steps", (1, 34))),
            test_time_steps=tuple(cfg.get("test_time_steps", (35, 49))),
            drop_unknown=cfg.get("drop_unknown", True),
        )

    if kind == "ibm_aml":
        return load_ibm_aml(
            root=cfg["root"],
            trans_file=cfg["trans_file"],
            val_fraction=cfg.get("val_fraction", 0.15),
            compute_betweenness=cfg.get("compute_betweenness", False),
            seed=seed,
        )

    raise ValueError(f"Tipo de dataset desconhecido: {kind!r}")


__all__ = ["GraphSample", "load_dataset", "load_elliptic", "load_ibm_aml"]