from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np

from .data import load_bag, shuffled
from .metrics import classification_metrics
from .utils import append_jsonl, ensure_dir


def run_torch_epoch(model, records, optimizer=None, device="cpu", max_patches=None, seed=0):
    import torch
    import torch.nn.functional as F

    training = optimizer is not None
    model.train(training)
    losses, labels, probabilities = [], [], []
    ordered = shuffled(records, seed) if training else records
    started = time.perf_counter()
    for index, record in enumerate(ordered):
        bag = load_bag(record, max_patches=max_patches, seed=seed + index)
        x = torch.from_numpy(bag).to(device)
        y = torch.tensor([record.label], dtype=torch.long, device=device)
        organ = torch.tensor([record.organ], dtype=torch.long, device=device)
        with torch.set_grad_enabled(training):
            output = model(x, organ)
            loss = F.cross_entropy(output["bag_logits"], y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        losses.append(float(loss.detach().cpu()))
        labels.append(record.label)
        probabilities.append(output["bag_prob"].detach().cpu().numpy()[0])
    metrics = classification_metrics(labels, probabilities)
    metrics.update(loss=float(np.mean(losses)), seconds=time.perf_counter() - started)
    return metrics


def run_jittor_epoch(model, records, optimizer=None, max_patches=None, seed=0):
    import jittor as jt
    from jittor import nn

    training = optimizer is not None
    model.train() if training else model.eval()
    losses, labels, probabilities = [], [], []
    ordered = shuffled(records, seed) if training else records
    started = time.perf_counter()
    context = jt.enable_grad() if training else jt.no_grad()
    with context:
        for index, record in enumerate(ordered):
            bag = load_bag(record, max_patches=max_patches, seed=seed + index)
            x = jt.array(bag).float32()
            y = jt.array([record.label]).int32()
            organ = jt.array([record.organ]).int32()
            output = model(x, organ)
            loss = nn.cross_entropy_loss(output["bag_logits"], y)
            if training:
                optimizer.step(loss)
            losses.append(float(loss.item()))
            labels.append(record.label)
            probabilities.append(output["bag_prob"].numpy()[0])
    jt.sync_all(True)
    metrics = classification_metrics(labels, probabilities)
    metrics.update(loss=float(np.mean(losses)), seconds=time.perf_counter() - started)
    return metrics


def fit(
    backend: str,
    model,
    optimizer,
    train_records,
    val_records,
    cfg: dict,
    checkpoint_path: Path,
    log_path: Path,
    device: str = "cpu",
    max_patches: int | None = None,
):
    best_loss = float("inf")
    stale = 0
    for epoch in range(int(cfg["epochs"])):
        if backend == "torch":
            train_metrics = run_torch_epoch(
                model, train_records, optimizer, device, max_patches, int(cfg["seed"]) + epoch
            )
            val_metrics = run_torch_epoch(model, val_records, None, device, max_patches, int(cfg["seed"]))
        else:
            train_metrics = run_jittor_epoch(
                model, train_records, optimizer, max_patches, int(cfg["seed"]) + epoch
            )
            val_metrics = run_jittor_epoch(model, val_records, None, max_patches, int(cfg["seed"]))
        record = {"epoch": epoch + 1, "train": train_metrics, "val": val_metrics}
        append_jsonl(log_path, record)
        print(json.dumps(record, ensure_ascii=False))
        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            stale = 0
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            if backend == "torch":
                import torch

                torch.save(model.state_dict(), checkpoint_path)
            else:
                model.save(str(checkpoint_path))
        else:
            stale += 1
            if stale >= int(cfg.get("patience", 20)):
                break
    return best_loss


def save_predictions(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

