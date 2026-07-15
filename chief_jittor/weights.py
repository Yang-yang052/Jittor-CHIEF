from __future__ import annotations

from pathlib import Path

import numpy as np


def torch_state_to_numpy(state: dict) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for key, value in state.items():
        clean = key.removeprefix("module.")
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        output[clean] = np.asarray(value)
    return output


def official_to_clean_names(state: dict[str, np.ndarray], dropout: bool = True) -> dict[str, np.ndarray]:
    """Map names from official models/CHIEF.py to this repository.

    Official attention_net is Sequential(linear, relu, [dropout], gated_attention).
    """
    gated_index = "3" if dropout else "2"
    mapped: dict[str, np.ndarray] = {}
    for key, value in state.items():
        new_key = key
        new_key = new_key.replace("attention_net.0.", "fc.")
        new_key = new_key.replace(f"attention_net.{gated_index}.", "gated_attention.")
        # Only the top-level official head is named ``classifiers``.  Do not
        # alter ``instance_classifiers`` (an auxiliary ModuleList).
        if new_key.startswith("classifiers."):
            new_key = "classifier." + new_key.removeprefix("classifiers.")
        # Tumor-origin checkpoints sometimes use the singular spelling already.
        mapped[new_key] = value
    return mapped


def save_npz(state: dict[str, np.ndarray], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **state)


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def load_into_jittor(model, state: dict[str, np.ndarray]) -> tuple[list[str], list[str]]:
    target = model.state_dict(to="numpy")
    matched = {key: value for key, value in state.items() if key in target and target[key].shape == value.shape}
    missing = sorted(set(target) - set(matched))
    unexpected = sorted(set(state) - set(matched))
    model.load_state_dict(matched)
    return missing, unexpected


def load_into_torch(model, state: dict[str, np.ndarray], strict: bool = False):
    import torch

    tensors = {key: torch.from_numpy(np.asarray(value)) for key, value in state.items()}
    return model.load_state_dict(tensors, strict=strict)
