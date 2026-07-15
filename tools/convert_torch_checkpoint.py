from __future__ import annotations

import argparse

import torch

from chief_jittor.weights import official_to_clean_names, save_npz, torch_state_to_numpy


def main():
    parser = argparse.ArgumentParser(description="Convert official CHIEF PyTorch weights to portable NPZ.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--no-dropout", action="store_true")
    args = parser.parse_args()
    state = torch.load(args.input, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = official_to_clean_names(torch_state_to_numpy(state), dropout=not args.no_dropout)
    save_npz(state, args.output)
    print(f"Saved {len(state)} arrays to {args.output}")


if __name__ == "__main__":
    main()

