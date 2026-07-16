from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def resolved_config(base: dict, seed: int) -> tuple[dict, Path]:
    data_dir = Path("data") / "hard" / f"seed_{seed}"
    log_dir = Path("logs") / "hard" / f"seed_{seed}"
    result_dir = Path("results") / "hard" / f"seed_{seed}"
    checkpoint_dir = Path("checkpoints") / "hard" / f"seed_{seed}"
    cfg = dict(base)
    cfg.pop("experiment_seeds", None)
    cfg.pop("data_generation", None)
    cfg.update(
        seed=seed,
        data_root=str(data_dir / "features"),
        train_csv=str(data_dir / "train.csv"),
        val_csv=str(data_dir / "val.csv"),
        test_csv=str(data_dir / "test.csv"),
        checkpoint_dir=str(checkpoint_dir),
        log_dir=str(log_dir),
        result_dir=str(result_dir),
    )
    config_path = ROOT / log_dir / "resolved_config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return cfg, config_path


def prepare_data(base: dict, seed: int) -> None:
    generation = base["data_generation"]
    output = Path("data") / "hard" / f"seed_{seed}"
    command = [
        sys.executable,
        "scripts/make_toy_data.py",
        "--output",
        str(output),
        "--feature-dim",
        str(base.get("input_dim", 768)),
        "--classes",
        str(base["n_classes"]),
        "--train",
        str(generation["train"]),
        "--val",
        str(generation["val"]),
        "--test",
        str(generation["test"]),
        "--seed",
        str(seed),
        "--min-patches",
        str(generation["min_patches"]),
        "--max-patches",
        str(generation["max_patches"]),
        "--background-std",
        str(generation["background_std"]),
        "--informative-fraction",
        str(generation["informative_fraction"]),
        "--signal-strength",
        str(generation["signal_strength"]),
        "--bag-shift-std",
        str(generation["bag_shift_std"]),
        "--train-label-noise",
        str(generation["train_label_noise"]),
        "--test-domain-shift",
        str(generation["test_domain_shift"]),
        "--organ-index",
        str(generation["organ_index"]),
    ]
    run(command)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the three-seed difficult synthetic CHIEF experiment.")
    parser.add_argument("--config", default="configs/hard.yaml")
    parser.add_argument("--backend", choices=["torch", "jittor"], required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--seeds", type=int, nargs="*")
    parser.add_argument("--skip-data", action="store_true")
    args = parser.parse_args()

    base_path = ROOT / args.config
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    seeds = args.seeds or [int(value) for value in base["experiment_seeds"]]

    for seed in seeds:
        _, config_path = resolved_config(base, seed)
        relative_config = str(config_path.relative_to(ROOT))
        if not args.skip_data:
            prepare_data(base, seed)
        init_path = Path("logs") / "hard" / f"seed_{seed}" / "common_init.npz"
        if not (ROOT / init_path).exists():
            run(
                [
                    sys.executable,
                    "tools/create_common_init.py",
                    "--config",
                    relative_config,
                    "--output",
                    str(init_path),
                ]
            )
        run(
            [
                sys.executable,
                "train.py",
                "--config",
                relative_config,
                "--backend",
                args.backend,
                "--device",
                args.device,
                "--init-weights",
                str(init_path),
            ]
        )
        run(
            [
                sys.executable,
                "test.py",
                "--config",
                relative_config,
                "--backend",
                args.backend,
                "--device",
                args.device,
            ]
        )

    run(
        [
            sys.executable,
            "tools/summarize_hard_experiment.py",
            "--config",
            args.config,
        ]
    )


if __name__ == "__main__":
    main()
