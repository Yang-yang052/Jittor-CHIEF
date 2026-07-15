from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

os.environ.setdefault("JITTOR_HOME", str(Path(".jittor-cache").resolve()))
if os.environ.get("CHIEF_USE_CUDA", "0") != "1":
    os.environ["nvcc_path"] = ""
import jittor as jt
from jittor import nn

from chief_jittor.model import CHIEFJittor
from chief_jittor.torch_reference import CHIEFTorch
from chief_jittor.weights import load_into_jittor, torch_state_to_numpy


def difference(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    delta = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return {"max_abs": float(delta.max()), "mean_abs": float(delta.mean())}


def main():
    parser = argparse.ArgumentParser(description="Forward/loss/one-step alignment for PyTorch and Jittor CHIEF.")
    parser.add_argument("--size", choices=["xs", "small"], default="small")
    parser.add_argument("--classes", type=int, default=3)
    parser.add_argument("--patches", type=int, default=23)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", default="logs/alignment.json")
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    jt.flags.use_cuda = 0
    jt.set_global_seed(args.seed)
    kwargs = dict(size_arg=args.size, dropout=False, n_classes=args.classes, use_organ_context=True)
    torch_model = CHIEFTorch(**kwargs).cpu().eval()
    jittor_model = CHIEFJittor(**kwargs).eval()
    # Convert through NumPy so the alignment does not depend on cross-framework
    # tensor conversion behavior.
    missing, unexpected = load_into_jittor(jittor_model, torch_state_to_numpy(torch_model.state_dict()))
    if missing or unexpected:
        raise RuntimeError({"missing": missing, "unexpected": unexpected})
    feature_dim = torch_model.input_dim
    features_np = np.random.default_rng(args.seed).normal(size=(args.patches, feature_dim)).astype(np.float32)
    organ_np = np.asarray([2], dtype=np.int32)
    label_np = np.asarray([1], dtype=np.int64)
    with torch.no_grad():
        torch_out = torch_model(torch.from_numpy(features_np), torch.from_numpy(organ_np).long())
    with jt.no_grad():
        jt_out = jittor_model(jt.array(features_np), jt.array(organ_np))
    report = {"forward": {}}
    for key in ("bag_logits", "bag_prob", "attention_raw", "attention", "WSI_feature", "WSI_feature_anatomical"):
        report["forward"][key] = difference(torch_out[key].numpy(), jt_out[key].numpy())

    torch_model.train()
    jittor_model.train()
    # Dropout is disabled, so both training graphs remain deterministic.
    torch_optimizer = torch.optim.SGD(torch_model.parameters(), lr=1e-4)
    jt_optimizer = jt.optim.SGD(jittor_model.parameters(), lr=1e-4)
    torch_logits = torch_model(torch.from_numpy(features_np), torch.from_numpy(organ_np).long())["bag_logits"]
    torch_loss = F.cross_entropy(torch_logits, torch.from_numpy(label_np).long())
    torch_optimizer.zero_grad(set_to_none=True)
    torch_loss.backward()
    torch_optimizer.step()
    jt_logits = jittor_model(jt.array(features_np), jt.array(organ_np))["bag_logits"]
    jt_loss = nn.cross_entropy_loss(jt_logits, jt.array(label_np).int32())
    jt_optimizer.step(jt_loss)
    jt.sync_all(True)
    report["loss"] = {"torch": float(torch_loss.detach()), "jittor": float(jt_loss.item())}
    report["loss"]["abs"] = abs(report["loss"]["torch"] - report["loss"]["jittor"])
    torch_fc = torch_model.fc.weight.detach().numpy()
    jt_fc = jittor_model.fc.weight.numpy()
    report["one_step_fc_weight"] = difference(torch_fc, jt_fc)
    report["pass"] = bool(
        max(item["max_abs"] for item in report["forward"].values()) < 1e-5
        and report["loss"]["abs"] < 1e-5
        and report["one_step_fc_weight"]["max_abs"] < 2e-5
    )
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["pass"]:
        raise SystemExit("Alignment thresholds were not met")


if __name__ == "__main__":
    main()
