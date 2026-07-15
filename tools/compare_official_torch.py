from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch
from torch import nn

from chief_jittor.torch_reference import CHIEFTorch
from chief_jittor.weights import load_into_torch, official_to_clean_names, torch_state_to_numpy


def initialize_weights(module):
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)


def load_official_class(path: Path):
    fake_package = types.ModuleType("utils")
    fake_utils = types.ModuleType("utils.utils")
    fake_utils.initialize_weights = initialize_weights
    sys.modules["utils"] = fake_package
    sys.modules["utils.utils"] = fake_utils
    spec = importlib.util.spec_from_file_location("official_chief", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module.CHIEF


def main():
    parser = argparse.ArgumentParser(description="Compare this PyTorch reference with official models/CHIEF.py.")
    parser.add_argument("--official", required=True, help="Path to the cloned official CHIEF repository")
    parser.add_argument("--output", default="logs/official_torch_equivalence.json")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    official_class = load_official_class(Path(args.official) / "models" / "CHIEF.py")
    embedding = torch.randn(19, 768)
    original_load = torch.load
    torch.load = lambda *unused_args, **unused_kwargs: embedding.clone()
    try:
        official = official_class(size_arg="small", dropout=True, n_classes=3).eval()
    finally:
        torch.load = original_load
    clean = CHIEFTorch(
        size_arg="small", dropout=True, n_classes=3,
        use_organ_context=True, organ_embedding=embedding,
    ).eval()
    mapped = official_to_clean_names(torch_state_to_numpy(official.state_dict()), dropout=True)
    incompatible = load_into_torch(clean, mapped, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError({"missing": incompatible.missing_keys, "unexpected": incompatible.unexpected_keys})
    features = torch.randn(41, 768)
    organ = torch.tensor([4])
    with torch.no_grad():
        official_output = official(features, organ)
        clean_output = clean(features, organ)
    report = {}
    for key in ("bag_logits", "attention_raw", "WSI_feature", "WSI_feature_anatomical"):
        delta = (official_output[key] - clean_output[key]).abs().numpy()
        report[key] = {"max_abs": float(delta.max()), "mean_abs": float(delta.mean())}
    report["pass"] = max(item["max_abs"] for item in report.values()) < 1e-6
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["pass"]:
        raise SystemExit("Official PyTorch equivalence threshold was not met")


if __name__ == "__main__":
    main()
