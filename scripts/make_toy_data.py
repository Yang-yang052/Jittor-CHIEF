from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def normalized_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def build_split(
    root: Path,
    split: str,
    count: int,
    rng: np.random.Generator,
    prototypes: np.ndarray,
    feature_dim: int,
    *,
    min_patches: int,
    max_patches: int,
    background_std: float,
    informative_fraction: float,
    signal_strength: float,
    bag_shift_std: float,
    label_noise: float,
    organ_index: int,
):
    rows = []
    feature_root = root / "features"
    feature_root.mkdir(parents=True, exist_ok=True)
    n_classes = len(prototypes)
    for index in range(count):
        true_label = index % n_classes
        label = true_label
        is_noisy = False
        if label_noise > 0 and rng.random() < label_noise:
            alternatives = [value for value in range(n_classes) if value != true_label]
            label = int(rng.choice(alternatives))
            is_noisy = True
        case_id = f"{split}_{index:04d}"
        patch_count = int(rng.integers(min_patches, max_patches + 1))
        features = rng.normal(0, background_std, (patch_count, feature_dim)).astype(np.float32)
        if bag_shift_std > 0:
            bag_shift = rng.normal(size=feature_dim).astype(np.float32)
            bag_shift /= max(float(np.linalg.norm(bag_shift)), 1e-12)
            features += bag_shift * bag_shift_std
        informative_count = max(3, int(round(patch_count * informative_fraction)))
        informative_count = min(informative_count, patch_count)
        informative = rng.choice(patch_count, informative_count, replace=False)
        features[informative] += prototypes[true_label] * signal_strength
        np.save(feature_root / f"{case_id}.npy", features)
        resolved_organ = true_label % 19 if organ_index < 0 else organ_index
        rows.append(
            {
                "case_id": case_id,
                "label": label,
                "true_label": true_label,
                "is_noisy": int(is_noisy),
                "organ": resolved_organ,
                "patch_count": patch_count,
                "informative_count": informative_count,
            }
        )
    pd.DataFrame(rows).to_csv(root / f"{split}.csv", index=False)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Generate deterministic CHIEF feature bags for smoke tests.")
    parser.add_argument("--output", default="data/toy")
    parser.add_argument("--feature-dim", type=int, default=768)
    parser.add_argument("--classes", type=int, default=3)
    parser.add_argument("--train", type=int, default=36)
    parser.add_argument("--val", type=int, default=12)
    parser.add_argument("--test", type=int, default=12)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--min-patches", type=int, default=48)
    parser.add_argument("--max-patches", type=int, default=127)
    parser.add_argument("--background-std", type=float, default=0.7)
    parser.add_argument("--informative-fraction", type=float, default=0.25)
    parser.add_argument("--signal-strength", type=float, default=2.5)
    parser.add_argument("--bag-shift-std", type=float, default=0.0)
    parser.add_argument("--train-label-noise", type=float, default=0.0)
    parser.add_argument("--test-domain-shift", type=float, default=0.0)
    parser.add_argument(
        "--organ-index",
        type=int,
        default=-1,
        help="Fixed organ index. Negative keeps the original label modulo 19 behavior.",
    )
    args = parser.parse_args()
    if not 0 < args.informative_fraction <= 1:
        raise SystemExit("--informative-fraction must be in (0, 1]")
    if not 0 <= args.train_label_noise < 1:
        raise SystemExit("--train-label-noise must be in [0, 1)")
    if not 0 <= args.test_domain_shift <= 1:
        raise SystemExit("--test-domain-shift must be in [0, 1]")
    if args.min_patches < 1 or args.max_patches < args.min_patches:
        raise SystemExit("Patch limits must satisfy 1 <= min_patches <= max_patches")

    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    prototypes = normalized_rows(rng.normal(size=(args.classes, args.feature_dim)).astype(np.float32))
    shifted = prototypes
    if args.test_domain_shift > 0:
        shift_directions = normalized_rows(
            rng.normal(size=(args.classes, args.feature_dim)).astype(np.float32)
        )
        shifted = normalized_rows(
            (1.0 - args.test_domain_shift) * prototypes
            + args.test_domain_shift * shift_directions
        )
    split_rows = {}
    for split, count in (("train", args.train), ("val", args.val), ("test", args.test)):
        split_rows[split] = build_split(
            root,
            split,
            count,
            rng,
            shifted if split == "test" else prototypes,
            args.feature_dim,
            min_patches=args.min_patches,
            max_patches=args.max_patches,
            background_std=args.background_std,
            informative_fraction=args.informative_fraction,
            signal_strength=args.signal_strength,
            bag_shift_std=args.bag_shift_std,
            label_noise=args.train_label_noise if split == "train" else 0.0,
            organ_index=args.organ_index,
        )
    metadata = {
        "seed": args.seed,
        "feature_dim": args.feature_dim,
        "classes": args.classes,
        "counts": {"train": args.train, "val": args.val, "test": args.test},
        "min_patches": args.min_patches,
        "max_patches": args.max_patches,
        "background_std": args.background_std,
        "informative_fraction": args.informative_fraction,
        "signal_strength": args.signal_strength,
        "bag_shift_std": args.bag_shift_std,
        "train_label_noise": args.train_label_noise,
        "realized_train_noisy_labels": int(sum(row["is_noisy"] for row in split_rows["train"])),
        "test_domain_shift": args.test_domain_shift,
        "organ_index": args.organ_index,
    }
    (root / "generation.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Toy dataset written to {root.resolve()}")
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
