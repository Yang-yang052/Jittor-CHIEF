from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str((Path(".cache") / "matplotlib").resolve()))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main():
    parser = argparse.ArgumentParser(description="Plot PyTorch/Jittor CHIEF loss alignment curves.")
    parser.add_argument("--torch-log", default="logs/toy/train_torch.jsonl")
    parser.add_argument("--jittor-log", default="logs/toy/train_jittor.jsonl")
    parser.add_argument("--output", default="results/toy/loss_alignment.png")
    args = parser.parse_args()
    plt.figure(figsize=(7.2, 4.5))
    for name, path, color in (("PyTorch", args.torch_log, "#2878B5"), ("Jittor", args.jittor_log, "#D95F02")):
        if not Path(path).exists():
            continue
        rows = read(path)
        epochs = [row["epoch"] for row in rows]
        plt.plot(epochs, [row["train"]["loss"] for row in rows], color=color, label=f"{name} train")
        plt.plot(epochs, [row["val"]["loss"] for row in rows], color=color, linestyle="--", label=f"{name} val")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title("CHIEF small-data training alignment")
    plt.grid(alpha=0.2)
    plt.legend()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    print(output.resolve())


if __name__ == "__main__":
    main()
