from __future__ import annotations

import argparse
import os
import platform
import sys
from pathlib import Path

import numpy as np

from chief_jittor.config import load_config
from chief_jittor.data import load_records
from chief_jittor.engine import fit
from chief_jittor.utils import ensure_dir, seed_everything
from chief_jittor.weights import load_into_jittor, load_into_torch, load_npz


def main():
    parser = argparse.ArgumentParser(description="Train CHIEF WSI MIL model with PyTorch or Jittor.")
    parser.add_argument("--config", default="configs/toy.yaml")
    parser.add_argument("--backend", choices=["torch", "jittor"], default="jittor")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--init-weights", help="Portable NPZ converted from an official PyTorch checkpoint")
    parser.add_argument("--max-patches", type=int)
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed_everything(int(cfg["seed"]))
    train_records = load_records(cfg["train_csv"], cfg["data_root"])
    val_records = load_records(cfg["val_csv"], cfg["data_root"])
    checkpoint_dir = ensure_dir(cfg["checkpoint_dir"])
    log_dir = ensure_dir(cfg["log_dir"])
    checkpoint = checkpoint_dir / f"chief_{args.backend}.{'pth' if args.backend == 'torch' else 'pkl'}"
    log_path = log_dir / f"train_{args.backend}.jsonl"
    if log_path.exists():
        log_path.unlink()

    model_kwargs = dict(
        size_arg=cfg["size_arg"],
        dropout=bool(cfg["dropout"]),
        n_classes=int(cfg["n_classes"]),
        n_organs=int(cfg["n_organs"]),
        organ_dim=int(cfg["organ_dim"]),
        use_organ_context=bool(cfg["use_organ_context"]),
    )
    if args.backend == "torch":
        import torch

        from chief_jittor.torch_reference import CHIEFTorch

        torch.manual_seed(int(cfg["seed"]))
        if args.device == "cuda" and not torch.cuda.is_available():
            raise SystemExit("PyTorch CUDA is unavailable. Install a CUDA-enabled PyTorch wheel or use --device cpu.")
        model = CHIEFTorch(**model_kwargs).to(args.device)
        if args.init_weights:
            print(load_into_torch(model, load_npz(args.init_weights), strict=False))
        optimizer = torch.optim.Adam(
            model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"])
        )
    else:
        os.environ.setdefault("JITTOR_HOME", str(Path(".jittor-cache").resolve()))
        import jittor as jt

        from chief_jittor.model import CHIEFJittor

        jt.flags.use_cuda = int(args.device == "cuda")
        jt.set_global_seed(int(cfg["seed"]))
        model = CHIEFJittor(**model_kwargs)
        if args.init_weights:
            missing, unexpected = load_into_jittor(model, load_npz(args.init_weights))
            print({"missing": missing, "unexpected": unexpected})
        optimizer = jt.optim.Adam(
            model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"])
        )

    best = fit(
        args.backend,
        model,
        optimizer,
        train_records,
        val_records,
        cfg,
        checkpoint,
        log_path,
        args.device,
        args.max_patches,
    )
    runtime_path = log_dir / f"environment_{args.backend}.txt"
    runtime_path.write_text(
        f"backend={args.backend}\npython={sys.version}\nplatform={platform.platform()}\ndevice={args.device}\nbest_val_loss={best}\n",
        encoding="utf-8",
    )
    print(f"Best validation loss: {best:.6f}; checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()

