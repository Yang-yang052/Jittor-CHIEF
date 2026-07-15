from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Convert official CHIEF .pt patch bags to framework-neutral .npy.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    input_root, output_root = Path(args.input), Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    paths = sorted(input_root.glob("*.pt"))
    if not paths:
        raise SystemExit(f"No .pt files found under {input_root}")
    for path in tqdm(paths):
        value = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(value, dict):
            for key in ("features", "feature", "data"):
                if key in value:
                    value = value[key]
                    break
        array = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
        np.save(output_root / f"{path.stem}.npy", array.astype(np.float32))
    print(f"Converted {len(paths)} bags to {output_root.resolve()}")


if __name__ == "__main__":
    main()

