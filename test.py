from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from chief_jittor.config import load_config
from chief_jittor.data import load_bag, load_records
from chief_jittor.metrics import classification_metrics
from chief_jittor.utils import ensure_dir


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained CHIEF model and export attention scores.")
    parser.add_argument("--config", default="configs/toy.yaml")
    parser.add_argument("--backend", choices=["torch", "jittor"], default="jittor")
    parser.add_argument("--checkpoint")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--max-patches", type=int)
    args = parser.parse_args()
    cfg = load_config(args.config)
    records = load_records(cfg["test_csv"], cfg["data_root"])
    result_dir = ensure_dir(cfg["result_dir"])
    attention_dir = ensure_dir(result_dir / f"attention_{args.backend}")
    checkpoint = args.checkpoint or str(
        Path(cfg["checkpoint_dir"]) / f"chief_{args.backend}.{'pth' if args.backend == 'torch' else 'pkl'}"
    )
    kwargs = dict(
        size_arg=cfg["size_arg"], dropout=bool(cfg["dropout"]), n_classes=int(cfg["n_classes"]),
        n_organs=int(cfg["n_organs"]), organ_dim=int(cfg["organ_dim"]),
        use_organ_context=bool(cfg["use_organ_context"]),
    )
    if args.backend == "torch":
        import torch
        from chief_jittor.torch_reference import CHIEFTorch

        model = CHIEFTorch(**kwargs).to(args.device)
        model.load_state_dict(torch.load(checkpoint, map_location=args.device, weights_only=False))
        model.eval()

        def predict(bag, organ):
            with torch.no_grad():
                output = model(
                    torch.from_numpy(bag).to(args.device),
                    torch.tensor([organ], dtype=torch.long, device=args.device),
                )
            return {key: value.detach().cpu().numpy() for key, value in output.items()}
    else:
        os.environ.setdefault("JITTOR_HOME", str(Path(".jittor-cache").resolve()))
        import jittor as jt
        from chief_jittor.model import CHIEFJittor

        jt.flags.use_cuda = int(args.device == "cuda")
        model = CHIEFJittor(**kwargs)
        model.load(checkpoint)
        model.eval()

        def predict(bag, organ):
            with jt.no_grad():
                output = model(jt.array(bag).float32(), jt.array([organ]).int32())
            return {key: value.numpy() for key, value in output.items()}

    rows, labels, probabilities = [], [], []
    for index, record in enumerate(records):
        bag = load_bag(record, args.max_patches, int(cfg["seed"]) + index)
        output = predict(bag, record.organ)
        prob = output["bag_prob"][0]
        pred = int(prob.argmax())
        labels.append(record.label)
        probabilities.append(prob)
        row = {"case_id": record.case_id, "label": record.label, "prediction": pred}
        row.update({f"prob_{i}": float(value) for i, value in enumerate(prob)})
        rows.append(row)
        np.savez_compressed(
            attention_dir / f"{record.case_id}.npz",
            attention=output["attention"][0],
            attention_raw=output["attention_raw"][0],
        )
    metrics = classification_metrics(labels, probabilities)
    pd.DataFrame(rows).to_csv(result_dir / f"predictions_{args.backend}.csv", index=False)
    (result_dir / f"metrics_{args.backend}.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

