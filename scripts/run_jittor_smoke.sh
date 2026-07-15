#!/usr/bin/env bash
set -euo pipefail

python scripts/make_toy_data.py --output data/toy --feature-dim 768 --train 36 --val 12 --test 12
python tools/create_common_init.py --config configs/toy.yaml --output logs/toy/common_init.npz
python tools/align_torch_jittor.py --size small --patches 23 --output logs/alignment_small.json
python train.py --config configs/toy.yaml --backend jittor --device cpu --init-weights logs/toy/common_init.npz
python test.py --config configs/toy.yaml --backend jittor --device cpu
python tools/benchmark.py --backend jittor --device cpu --patches 512 --runs 30 --output logs/performance.jsonl
python tools/plot_logs.py --output results/toy/loss_alignment.png

echo "Jittor CHIEF smoke reproduction completed."
