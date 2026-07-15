from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Measure CHIEF WSI inference latency.")
    parser.add_argument("--backend", choices=["torch", "jittor"], required=True)
    parser.add_argument("--patches", type=int, default=512)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--output", default="logs/performance.jsonl")
    args = parser.parse_args()
    rng = np.random.default_rng(2026)
    features = rng.normal(size=(args.patches, 768)).astype(np.float32)
    if args.backend == "torch":
        import torch
        from chief_jittor.torch_reference import CHIEFTorch

        model = CHIEFTorch(size_arg="small", dropout=False, n_classes=3, use_organ_context=False).to(args.device).eval()
        x = torch.from_numpy(features).to(args.device)
        params = sum(p.numel() for p in model.parameters())
        def call():
            with torch.no_grad():
                model(x)
            if args.device == "cuda":
                torch.cuda.synchronize()
    else:
        os.environ.setdefault("JITTOR_HOME", str(Path(".jittor-cache").resolve()))
        import jittor as jt
        from chief_jittor.model import CHIEFJittor

        jt.flags.use_cuda = int(args.device == "cuda")
        model = CHIEFJittor(size_arg="small", dropout=False, n_classes=3, use_organ_context=False).eval()
        x = jt.array(features)
        params = sum(int(np.prod(p.shape)) for p in model.parameters())
        def call():
            with jt.no_grad():
                model(x)
            jt.sync_all(True)
    for _ in range(5):
        call()
    times = []
    for _ in range(args.runs):
        start = time.perf_counter()
        call()
        times.append((time.perf_counter() - start) * 1000)
    report = {
        "backend": args.backend, "device": args.device, "patches": args.patches,
        "runs": args.runs, "parameters": params, "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)), "p95_ms": float(np.percentile(times, 95)),
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(report) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

