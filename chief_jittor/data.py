from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BagRecord:
    case_id: str
    label: int
    feature_path: Path
    organ: int = 0


def load_records(csv_path: str | Path, data_root: str | Path) -> list[BagRecord]:
    frame = pd.read_csv(csv_path)
    required = {"case_id", "label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")
    data_root = Path(data_root)
    records: list[BagRecord] = []
    for row in frame.itertuples(index=False):
        case_id = str(row.case_id)
        explicit = getattr(row, "feature_path", None)
        path = Path(explicit) if isinstance(explicit, str) and explicit else data_root / f"{case_id}.npy"
        records.append(
            BagRecord(
                case_id=case_id,
                label=int(row.label),
                feature_path=path,
                organ=int(getattr(row, "organ", 0)),
            )
        )
    return records


def load_bag(record: BagRecord, max_patches: int | None = None, seed: int = 0) -> np.ndarray:
    path = record.feature_path
    if not path.exists():
        raise FileNotFoundError(f"Feature bag not found: {path}")
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as payload:
            key = "features" if "features" in payload else payload.files[0]
            bag = payload[key]
    elif path.suffix == ".npy":
        bag = np.load(path, allow_pickle=False)
    else:
        raise ValueError(f"Jittor loader accepts .npy/.npz, got {path.suffix}. Run tools/convert_pt_features.py first.")
    bag = np.asarray(bag, dtype=np.float32)
    if bag.ndim == 3 and bag.shape[0] == 1:
        bag = bag[0]
    if bag.ndim != 2:
        raise ValueError(f"Expected [num_patches, feature_dim], got {bag.shape} for {path}")
    if max_patches and len(bag) > max_patches:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(len(bag), max_patches, replace=False))
        bag = bag[indices]
    return np.ascontiguousarray(bag)


def shuffled(records: list[BagRecord], seed: int) -> list[BagRecord]:
    order = np.arange(len(records))
    np.random.default_rng(seed).shuffle(order)
    return [records[index] for index in order]

