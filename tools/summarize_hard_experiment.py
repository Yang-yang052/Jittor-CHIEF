from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str((Path(".cache") / "matplotlib").resolve()))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml


METRICS = ("accuracy", "balanced_accuracy", "macro_f1", "macro_auroc")
COLORS = {"torch": "#2878B5", "jittor": "#D95F02"}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def mean_std(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(np.mean(array)), "std": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0}


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize difficult synthetic PyTorch/Jittor experiments.")
    parser.add_argument("--config", default="configs/hard.yaml")
    parser.add_argument("--output-dir", default="results/hard")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    seeds = [int(value) for value in cfg["experiment_seeds"]]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    per_seed: dict[str, dict] = {}
    summary: dict[str, dict] = {}
    csv_rows = []
    for seed in seeds:
        seed_key = str(seed)
        per_seed[seed_key] = {}
        for backend in ("torch", "jittor"):
            metric_path = output_dir / f"seed_{seed}" / f"metrics_{backend}.json"
            if not metric_path.exists():
                continue
            values = read_json(metric_path)
            per_seed[seed_key][backend] = values
            csv_rows.append({"seed": seed, "backend": backend, **values})

    for backend in ("torch", "jittor"):
        backend_rows = [
            per_seed[str(seed)][backend]
            for seed in seeds
            if backend in per_seed[str(seed)]
        ]
        if backend_rows:
            summary[backend] = {
                metric: mean_std([float(row[metric]) for row in backend_rows])
                for metric in METRICS
            }

    paired_differences = {}
    for metric in METRICS:
        diffs = [
            abs(
                float(per_seed[str(seed)]["torch"][metric])
                - float(per_seed[str(seed)]["jittor"][metric])
            )
            for seed in seeds
            if {"torch", "jittor"} <= set(per_seed[str(seed)])
        ]
        if diffs:
            paired_differences[metric] = {
                **mean_std(diffs),
                "max": float(max(diffs)),
            }

    loss_differences = {}
    for seed in seeds:
        torch_log = read_jsonl(Path("logs") / "hard" / f"seed_{seed}" / "train_torch.jsonl")
        jittor_log = read_jsonl(Path("logs") / "hard" / f"seed_{seed}" / "train_jittor.jsonl")
        if not torch_log or not jittor_log:
            continue
        epochs = min(len(torch_log), len(jittor_log))
        loss_differences[str(seed)] = {
            "epochs_compared": epochs,
            "max_train_abs": float(
                max(
                    abs(torch_log[i]["train"]["loss"] - jittor_log[i]["train"]["loss"])
                    for i in range(epochs)
                )
            ),
            "max_val_abs": float(
                max(
                    abs(torch_log[i]["val"]["loss"] - jittor_log[i]["val"]["loss"])
                    for i in range(epochs)
                )
            ),
        }

    payload = {
        "experiment": "difficult_synthetic_three_seed",
        "seeds": seeds,
        "data_generation": cfg["data_generation"],
        "per_seed": per_seed,
        "summary": summary,
        "paired_absolute_differences": paired_differences,
        "loss_differences": loss_differences,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if csv_rows:
        with (output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(csv_rows)

    available = [backend for backend in ("torch", "jittor") if backend in summary]
    if available:
        x = np.arange(len(METRICS))
        width = 0.34 if len(available) == 2 else 0.55
        plt.figure(figsize=(8.2, 4.8))
        for index, backend in enumerate(available):
            offset = (index - (len(available) - 1) / 2) * width
            means = [summary[backend][metric]["mean"] for metric in METRICS]
            stds = [summary[backend][metric]["std"] for metric in METRICS]
            plt.bar(
                x + offset,
                means,
                width,
                yerr=stds,
                capsize=4,
                color=COLORS[backend],
                alpha=0.86,
                label="PyTorch" if backend == "torch" else "Jittor",
            )
        plt.xticks(x, ["Accuracy", "Balanced Acc.", "Macro-F1", "Macro-AUROC"])
        plt.ylim(0, 1.05)
        plt.ylabel("Score (mean ± SD, 3 seeds)")
        plt.title("Difficult synthetic CHIEF experiment")
        plt.grid(axis="y", alpha=0.2)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "metrics_comparison.png", dpi=200)
        plt.close()

    plt.figure(figsize=(8.4, 5.0))
    plotted = False
    for backend in ("torch", "jittor"):
        for split, linestyle in (("train", "-"), ("val", "--")):
            curves = []
            for seed in seeds:
                rows = read_jsonl(Path("logs") / "hard" / f"seed_{seed}" / f"train_{backend}.jsonl")
                if rows:
                    curves.append(np.asarray([row[split]["loss"] for row in rows], dtype=np.float64))
            if not curves:
                continue
            length = min(map(len, curves))
            matrix = np.stack([curve[:length] for curve in curves])
            epochs = np.arange(1, length + 1)
            mean = matrix.mean(axis=0)
            std = matrix.std(axis=0, ddof=1) if len(matrix) > 1 else np.zeros_like(mean)
            label = f"{'PyTorch' if backend == 'torch' else 'Jittor'} {split}"
            plt.plot(epochs, mean, color=COLORS[backend], linestyle=linestyle, linewidth=2, label=label)
            plt.fill_between(epochs, mean - std, mean + std, color=COLORS[backend], alpha=0.10)
            plotted = True
    if plotted:
        plt.xlabel("Epoch")
        plt.ylabel("Cross-entropy loss")
        plt.title("Difficult synthetic training alignment (mean ± SD)")
        plt.grid(alpha=0.2)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "loss_alignment.png", dpi=200)
    plt.close()
    print(output_dir / "summary.json")


if __name__ == "__main__":
    main()
