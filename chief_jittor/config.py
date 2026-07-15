from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


SIZE_DICT = {
    "xs": [384, 256, 256],
    "small": [768, 512, 256],
    "big": [1024, 512, 384],
    "large": [2048, 1024, 512],
}


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    size = SIZE_DICT[cfg.get("size_arg", "small")]
    cfg.setdefault("input_dim", size[0])
    cfg.setdefault("hidden_dim", size[1])
    cfg.setdefault("attention_dim", size[2])
    cfg["config_path"] = str(path)
    return cfg

