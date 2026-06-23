"""
train.py (gnn)
==============
Treino de um estágio e orquestração do pipeline de transfer learning.

`run_pipeline` executa, na ordem do YAML:
  1. pré-treina o backbone no Elliptic           -> salva elliptic_backbone.pt
  2. carrega o backbone e faz fine-tuning no IBM HI-Small
  3. carrega o backbone e SÓ avalia no IBM LI-Small (proporção realista)

O encoder é sempre novo (in_dim do dataset); backbone + cabeça são transferidos.
Treino em mini-batch via NeighborLoader (GPU) ou full-batch (CPU / Elliptic).
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F

from datasets import load_dataset
from datasets.common import GraphSample

from .metrics import evaluate, format_metrics
from .model import GraphGuardGNN, load_backbone_into, save_backbone


# ----------------------------------------------------------------------
# Pesos de classe (fraude é minoria)
# ----------------------------------------------------------------------
def _class_weights(y: torch.Tensor, mask: torch.Tensor, mode: str) -> torch.Tensor | None:
    if mode != "balanced":
        return None
    yt = y[mask]
    counts = torch.bincount(yt, minlength=2).float()
    counts = torch.clamp(counts, min=1.0)
    w = counts.sum() / (2.0 * counts)        # n / (n_classes * count_c)
    return w.to(y.device)


# ----------------------------------------------------------------------
# Avaliação (full-batch — basta um forward em modo eval)
# ----------------------------------------------------------------------
@torch.no_grad()
def _evaluate(model, data, mask: torch.Tensor, max_fpr: float) -> dict:
    model.eval()
    logits = model(data.x, data.edge_index)
    proba = F.softmax(logits, dim=1)[:, 1]
    m = mask.cpu().numpy()
    return evaluate(data.y.cpu().numpy()[m], proba.cpu().numpy()[m], max_fpr=max_fpr)


# ----------------------------------------------------------------------
# Treino de UM estágio
# ----------------------------------------------------------------------
def train_stage(
    sample: GraphSample,
    model_cfg: dict,
    train_cfg: dict,
    epochs: int,
    lr: float,
    device: str,
    load_backbone: str | None = None,
    seed: int = 42,
) -> tuple[GraphGuardGNN, dict]:
    torch.manual_seed(seed)
    data = sample.to_pyg().to(device)

    model = GraphGuardGNN(
        in_dim=sample.in_dim,
        hidden_dim=model_cfg.get("hidden_dim", 64),
        num_layers=model_cfg.get("num_layers", 2),
        dropout=model_cfg.get("dropout", 0.3),
    ).to(device)

    if load_backbone:
        load_backbone_into(model, load_backbone)
        print(f"  backbone carregado de {os.path.basename(load_backbone)}")

    opt_cfg = train_cfg.get("optimizer", {})
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=opt_cfg.get("weight_decay", 5e-4),
    )
    weight = _class_weights(data.y, data.train_mask, train_cfg.get("class_weight", "balanced"))
    max_fpr = train_cfg.get("eval", {}).get("max_fpr", 0.05)

    ns_cfg = train_cfg.get("neighbor_sampling", {})
    use_sampling = ns_cfg.get("enabled", False) and int(data.train_mask.sum()) > 0

    if epochs > 0 and use_sampling:
        _train_minibatch(model, data, optimizer, weight, epochs, ns_cfg, max_fpr)
    elif epochs > 0:
        _train_fullbatch(model, data, optimizer, weight, epochs, max_fpr)
    else:
        print("  (estágio só de avaliação — sem treino)")

    metrics = _evaluate(model, data, data.eval_mask, max_fpr)
    return model, metrics


def _train_fullbatch(model, data, optimizer, weight, epochs, max_fpr):
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask], weight=weight)
        loss.backward()
        optimizer.step()
        if epoch % max(1, epochs // 10) == 0 or epoch == epochs:
            m = _evaluate(model, data, data.eval_mask, max_fpr)
            print(f"  época {epoch:3d}/{epochs}  loss={loss.item():.4f}  "
                  f"auroc={m['auroc']:.4f}  ap={m['ap']:.4f}")


def _train_minibatch(model, data, optimizer, weight, epochs, ns_cfg, max_fpr):
    from torch_geometric.loader import NeighborLoader

    loader = NeighborLoader(
        data,
        num_neighbors=ns_cfg.get("num_neighbors", [25, 10]),
        batch_size=ns_cfg.get("batch_size", 2048),
        input_nodes=data.train_mask,
        shuffle=True,
    )
    weight = weight.to(data.x.device) if weight is not None else None
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for batch in loader:
            optimizer.zero_grad()
            logits = model(batch.x, batch.edge_index)[: batch.batch_size]
            y = batch.y[: batch.batch_size]
            loss = F.cross_entropy(logits, y, weight=weight)
            loss.backward()
            optimizer.step()
            total += float(loss) * batch.batch_size
        if epoch % max(1, epochs // 10) == 0 or epoch == epochs:
            m = _evaluate(model, data, data.eval_mask, max_fpr)
            avg = total / max(1, int(data.train_mask.sum()))
            print(f"  época {epoch:3d}/{epochs}  loss={avg:.4f}  "
                  f"auroc={m['auroc']:.4f}  ap={m['ap']:.4f}")


# ----------------------------------------------------------------------
# Pipeline completo de transfer learning
# ----------------------------------------------------------------------
def run_pipeline(config: dict) -> dict:
    seed = config.get("seed", 42)
    device = config.get("device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        print("[aviso] cuda indisponível — caindo para cpu.")
        device = "cpu"

    ckpt_dir = config["paths"]["checkpoints"]
    os.makedirs(ckpt_dir, exist_ok=True)

    datasets_cfg = config["datasets"]
    model_cfg = config["model"]
    train_cfg = config["train"]

    results: dict[str, dict] = {}
    for stage in config["pipeline"]:
        name = stage["name"]
        ds_key = stage["dataset"]
        print("\n" + "=" * 70)
        print(f"  ESTÁGIO: {name}  (dataset: {ds_key})")
        print("=" * 70)

        sample = load_dataset(datasets_cfg[ds_key], seed=seed)
        print("  " + sample.summary())

        load_path = (
            os.path.join(ckpt_dir, stage["load_backbone"])
            if stage.get("load_backbone")
            else None
        )
        lr = stage.get("lr", train_cfg.get("optimizer", {}).get("lr", 0.005))

        model, metrics = train_stage(
            sample=sample,
            model_cfg=model_cfg,
            train_cfg=train_cfg,
            epochs=stage.get("epochs", 0),
            lr=lr,
            device=device,
            load_backbone=load_path,
            seed=seed,
        )
        print(f"  >> avaliação: {format_metrics(metrics)}")
        results[name] = metrics

        if stage.get("save_backbone"):
            save_path = os.path.join(ckpt_dir, stage["save_backbone"])
            save_backbone(model, save_path)
            print(f"  backbone salvo em {save_path}")

    return results