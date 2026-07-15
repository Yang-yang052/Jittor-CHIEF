from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str((Path(".cache") / "matplotlib").resolve()))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Visualize the most-attended patches for one WSI bag.")
    parser.add_argument("--input", required=True, help="NPZ exported by test.py")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--output", default="results/toy/attention_topk.png")
    args = parser.parse_args()
    with np.load(args.input) as payload:
        attention = payload["attention"].reshape(-1)
    indices = np.argsort(attention)[-args.top_k:][::-1]
    values = attention[indices]
    plt.figure(figsize=(8, 4.8))
    plt.bar(np.arange(len(values)), values, color="#2878B5")
    plt.xticks(np.arange(len(values)), indices, rotation=60)
    plt.xlabel("Patch index")
    plt.ylabel("Attention weight")
    plt.title(f"CHIEF top-{len(values)} attended patches")
    plt.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=180)
    print(output.resolve())


if __name__ == "__main__":
    main()
