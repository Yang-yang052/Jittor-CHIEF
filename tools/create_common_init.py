from __future__ import annotations

import argparse

import torch

from chief_jittor.config import load_config
from chief_jittor.torch_reference import CHIEFTorch
from chief_jittor.weights import save_npz, torch_state_to_numpy


def main():
    parser = argparse.ArgumentParser(description="Create one portable initialization for PyTorch/Jittor alignment.")
    parser.add_argument("--config", default="configs/toy.yaml")
    parser.add_argument("--output", default="logs/toy/common_init.npz")
    args = parser.parse_args()
    cfg = load_config(args.config)
    torch.manual_seed(int(cfg["seed"]))
    model = CHIEFTorch(
        size_arg=cfg["size_arg"],
        dropout=bool(cfg["dropout"]),
        n_classes=int(cfg["n_classes"]),
        n_organs=int(cfg["n_organs"]),
        organ_dim=int(cfg["organ_dim"]),
        use_organ_context=bool(cfg["use_organ_context"]),
    )
    state = torch_state_to_numpy(model.state_dict())
    save_npz(state, args.output)
    print(f"Saved {len(state)} arrays to {args.output}")


if __name__ == "__main__":
    main()
