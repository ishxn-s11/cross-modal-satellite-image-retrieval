"""IO, seeding, device and checkpoint helpers."""

from __future__ import annotations

import json
import os
import random
import sys
import time
from typing import Dict, Any, Optional, TextIO

import numpy as np
import torch


class Logger:
    """Logger that writes the same line to stdout and an optional log file."""

    def __init__(self, path: Optional[str] = None, stream: TextIO = sys.stdout) -> None:
        self.stream = stream
        if path:
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
        self.fh = open(path, "a", encoding="utf-8") if path else None

    def info(self, msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {msg}"
        print(line, file=self.stream, flush=True)
        if self.fh:
            self.fh.write(line + "\n")
            self.fh.flush()

    def close(self) -> None:
        if self.fh:
            self.fh.close()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(prefer: Optional[str] = None) -> torch.device:
    if prefer and prefer != "auto":
        return torch.device(prefer)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def num_params(model: torch.nn.Module, trainable_only: bool = True) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def save_checkpoint(model: torch.nn.Module, path: str, extra: Optional[Dict[str, Any]] = None) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {"model": model.state_dict(), "modalities": getattr(model, "modalities", None)}
    if extra:
        state.update(extra)
    torch.save(state, path)
    return path


def load_checkpoint(model: torch.nn.Module, path: str, device: torch.device) -> Dict[str, Any]:
    state = torch.load(path, map_location=device, weights_only=False)
    if "model" in state:  # checkpoints written by save_checkpoint
        model.load_state_dict(state["model"])
    elif "model_state_dict" in state:  # external/torchvision-style checkpoints
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    return state


def save_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=_json_default)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _json_default(o: Any):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


class Timer:
    """Simple context-manager / stopwatch for timing retrieval operations."""

    def __enter__(self) -> "Timer":
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.elapsed = time.perf_counter() - self.t0

    def ms(self) -> float:
        return getattr(self, "elapsed", 0.0) * 1000.0