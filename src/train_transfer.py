"""
train_transfer.py
=================
Entrypoint da segunda etapa: treina a GNN do GraphGuard por transfer learning
entre datasets reais de fraude financeira, guiado por um arquivo YAML.

Uso:
    # pré-treina no Elliptic -> fine-tune no IBM HI-Small -> valida no LI-Small
    python src/train_transfer.py --config configs/transfer.yaml

    # rodar só um estágio (ex.: validar) por nome
    python src/train_transfer.py --config configs/transfer.yaml --only validate_ibm_li_small

    # sobrescrever o device
    python src/train_transfer.py --config configs/transfer.yaml --device cpu

Pré-requisitos: torch + torch-geometric instalados e os CSVs do Kaggle
extraídos em data/raw/ (ver docs/SEGUNDA-ETAPA.md).
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yaml  # noqa: E402

from gnn.train import run_pipeline  # noqa: E402


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="GraphGuard — treino por transfer learning")
    parser.add_argument("--config", default="configs/transfer.yaml")
    parser.add_argument("--device", choices=["cuda", "cpu"], default=None,
                        help="sobrescreve config['device']")
    parser.add_argument("--only", default=None,
                        help="executa só o estágio com este 'name'")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.device:
        config["device"] = args.device
    if args.only:
        config["pipeline"] = [s for s in config["pipeline"] if s["name"] == args.only]
        if not config["pipeline"]:
            raise SystemExit(f"Estágio '{args.only}' não encontrado no config.")

    results = run_pipeline(config)

    print("\n" + "=" * 70)
    print("  RESUMO FINAL")
    print("=" * 70)
    for stage, m in results.items():
        auroc = m.get("auroc", float("nan"))
        ap = m.get("ap", float("nan"))
        rec_key = next((k for k in m if k.startswith("recall@")), None)
        rec = m.get(rec_key, float("nan")) if rec_key else float("nan")
        print(f"  {stage:<26} auroc={auroc:.4f}  ap={ap:.4f}  {rec_key or 'recall'}={rec:.4f}")


if __name__ == "__main__":
    main()