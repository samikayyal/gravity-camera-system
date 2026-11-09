"""Shared experiment settings and provenance checks."""

import importlib.metadata
import json
import platform
from pathlib import Path

import psutil
import torch

from prepare_dataset import ROOT, materialize, sha256

SEED = 42


def device_name(requested="auto"):
    return ("0" if torch.cuda.is_available() else "cpu") if requested == "auto" else requested


def environment():
    memory = psutil.Process().memory_info()
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_threads": torch.get_num_threads(),
        "cuda_available": torch.cuda.is_available(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in [
                "ultralytics",
                "torch",
                "torchvision",
                "numpy",
                "opencv-python",
                "psutil",
            ]
        },
        "process_peak_memory_bytes": getattr(memory, "peak_wset", None),
        "process_rss_bytes": memory.rss,
    }


def checked_manifest(data):
    data = Path(data).resolve()
    manifest = json.loads((data.parent / "manifest.json").read_text(encoding="utf-8"))
    materialize(manifest, ROOT / "data", data.parent)
    return manifest, sha256(data.parent / "manifest.json")


def fresh_output(path):
    path = Path(path).resolve()
    if path.exists():
        raise ValueError(f"Run output already exists; use a fresh --output: {path}")
    path.mkdir(parents=True)
    return path
