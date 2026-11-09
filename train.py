"""Run a bounded YOLO26n smoke test; longer training requires explicit --epochs."""

import argparse
import json
import time
from pathlib import Path

from ultralytics import YOLO

from experiment import SEED, checked_manifest, device_name, environment, fresh_output
from prepare_dataset import ROOT, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/fine-tuning/data.yaml")
    parser.add_argument("--model", type=Path, default=ROOT / "yolo26n.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/camera1-smoke")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if min(args.epochs, args.imgsz, args.batch) <= 0:
        parser.error("Epochs, image size, and batch size must be positive")
    _, manifest_hash = checked_manifest(args.data)
    model_hash = sha256(args.model)
    output = fresh_output(args.output)
    settings = {
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": 0,
        "device": device_name(args.device),
        "seed": SEED,
        "deterministic": True,
        "amp": False,
        "cache": False,
        "plots": False,
        "optimizer": "AdamW",
        "lr0": 0.001,
        "lrf": 0.01,
        "weight_decay": 0.0005,
        "warmup_epochs": 0.0,
        "close_mosaic": 0,
        "patience": 0,
        "save": True,
        "val": True,
        "rect": False,
    }
    write_json(
        output / "request.json",
        {
            "settings": settings,
            "environment": environment(),
            "manifest_sha256": manifest_hash,
            "model_sha256": model_hash,
        },
    )
    model = YOLO(str(args.model))
    if model.names.get(0) != "person":
        raise ValueError("Expected person at class 0")
    started = time.perf_counter()
    model.train(data=str(args.data.resolve()), project=str(output), name="training", **settings)
    elapsed = time.perf_counter() - started
    last = output / "training/weights/last.pt"
    best = output / "training/weights/best.pt"
    summary = {
        "settings": settings,
        "manifest_sha256": manifest_hash,
        "initial_model_sha256": model_hash,
        "environment": environment(),
        "wall_seconds": elapsed,
        "epochs_completed": model.trainer.epoch + 1,
        "comparison_checkpoint": "training/weights/last.pt",
        "checkpoint_sha256": {"last": sha256(last), "best": sha256(best)},
    }
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
