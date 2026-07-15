$ErrorActionPreference = "Stop"
$python = if (Test-Path "..\.venv\Scripts\python.exe") { "..\.venv\Scripts\python.exe" } else { "python" }

& $python scripts\make_toy_data.py --output data\toy --feature-dim 768 --train 36 --val 12 --test 12
& $python tools\create_common_init.py --config configs\toy.yaml --output logs\toy\common_init.npz
& $python train.py --config configs\toy.yaml --backend torch --device cpu --init-weights logs\toy\common_init.npz
& $python test.py --config configs\toy.yaml --backend torch --device cpu
& $python tools\benchmark.py --backend torch --device cpu --patches 512 --runs 30 --output logs\performance.jsonl
& $python tools\plot_logs.py --torch-log logs\toy\train_torch.jsonl --output results\toy\loss_curve_torch.png

Write-Host "PyTorch CHIEF smoke reproduction completed."
