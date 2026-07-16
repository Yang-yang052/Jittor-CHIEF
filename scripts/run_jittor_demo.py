from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(label: str, arguments: list[str]) -> None:
    command = [sys.executable, *arguments]
    print(f"\n[DEMO] {label}")
    print(" ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the complete Jittor-CHIEF toy training and testing demonstration."
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument(
        "--skip-data",
        action="store_true",
        help="Reuse the existing toy data and common initialization.",
    )
    args = parser.parse_args()

    if not args.skip_data:
        run(
            "1/7 Generate deterministic toy feature bags",
            [
                "scripts/make_toy_data.py",
                "--output",
                "data/toy",
                "--feature-dim",
                "768",
                "--train",
                "36",
                "--val",
                "12",
                "--test",
                "12",
                "--seed",
                "2026",
            ],
        )
        run(
            "2/7 Create the shared PyTorch/Jittor initialization",
            [
                "tools/create_common_init.py",
                "--config",
                "configs/toy.yaml",
                "--output",
                "logs/toy/common_init.npz",
            ],
        )

    run(
        "3/7 Check forward, loss and one-step cross-framework alignment",
        [
            "tools/align_torch_jittor.py",
            "--size",
            "small",
            "--patches",
            "23",
            "--output",
            "logs/alignment_demo.json",
        ],
    )
    run(
        "4/7 Train the Jittor implementation",
        [
            "train.py",
            "--config",
            "configs/toy.yaml",
            "--backend",
            "jittor",
            "--device",
            args.device,
            "--init-weights",
            "logs/toy/common_init.npz",
        ],
    )
    run(
        "5/7 Test and export metrics, predictions and attention",
        [
            "test.py",
            "--config",
            "configs/toy.yaml",
            "--backend",
            "jittor",
            "--device",
            args.device,
        ],
    )
    run(
        "6/7 Plot PyTorch/Jittor loss alignment",
        [
            "tools/plot_logs.py",
            "--output",
            "results/toy/loss_alignment.png",
        ],
    )
    run(
        "7/7 Visualize top attended patches",
        [
            "tools/visualize_attention.py",
            "--input",
            "results/toy/attention_jittor/test_0000.npz",
            "--output",
            "results/toy/attention_top20_jittor.png",
        ],
    )

    print("\n[DEMO] Complete. Open these files during the recording:")
    for output in [
        "logs/alignment_demo.json",
        "logs/toy/train_jittor.jsonl",
        "checkpoints/toy/chief_jittor.pkl",
        "results/toy/metrics_jittor.json",
        "results/toy/predictions_jittor.csv",
        "results/toy/attention_jittor/test_0000.npz",
        "results/toy/loss_alignment.png",
        "results/toy/attention_top20_jittor.png",
    ]:
        print(f"  - {output}")


if __name__ == "__main__":
    main()
