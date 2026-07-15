from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def build_split(
    root: Path,
    split: str,
    count: int,
    rng: np.random.Generator,
    prototypes: np.ndarray,
    feature_dim: int,
):
    rows = []
    feature_root = root / "features"
    feature_root.mkdir(parents=True, exist_ok=True)
    n_classes = len(prototypes)
    for index in range(count):
        label = index % n_classes
        case_id = f"{split}_{index:04d}"
        patch_count = int(rng.integers(48, 128))
        features = rng.normal(0, 0.7, (patch_count, feature_dim)).astype(np.float32)
        informative = rng.choice(patch_count, max(6, patch_count // 4), replace=False)
        features[informative] += prototypes[label] * 2.5
        np.save(feature_root / f"{case_id}.npy", features)
        rows.append({"case_id": case_id, "label": label, "organ": label % 19})
    pd.DataFrame(rows).to_csv(root / f"{split}.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description="Generate deterministic CHIEF feature bags for smoke tests.")
    parser.add_argument("--output", default="data/toy")
    parser.add_argument("--feature-dim", type=int, default=768)
    parser.add_argument("--classes", type=int, default=3)
    parser.add_argument("--train", type=int, default=36)
    parser.add_argument("--val", type=int, default=12)
    parser.add_argument("--test", type=int, default=12)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    prototypes = rng.normal(size=(args.classes, args.feature_dim)).astype(np.float32)
    prototypes /= np.linalg.norm(prototypes, axis=1, keepdims=True)
    for split, count in (("train", args.train), ("val", args.val), ("test", args.test)):
        build_split(root, split, count, rng, prototypes, args.feature_dim)
    print(f"Toy dataset written to {root.resolve()}")


if __name__ == "__main__":
    main()

