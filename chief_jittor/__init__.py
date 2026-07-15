"""Jittor implementation and reproducibility utilities for CHIEF.

Jittor's Windows import path automatically downloads a CUDA toolkit whenever a
GPU driver is detected.  The reproducibility scripts default to CPU so a fresh
clone remains lightweight.  Set ``CHIEF_USE_CUDA=1`` before Python starts to
opt into Jittor's CUDA discovery.
"""

import os

if os.environ.get("CHIEF_USE_CUDA", "0") != "1":
    os.environ["nvcc_path"] = ""

__version__ = "0.1.0"
