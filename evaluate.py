"""Comparable person-only validation, fixed-threshold errors, and warmed-up timing."""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionValidator

from experiment import checked_manifest, device_name, environment, fresh_output
from prepare_dataset import ROOT, sha256, write_json


class PersonValidator(DetectionValidator):
    """The installed upstream validator does not pass classes to postprocess NMS."""

    def postprocess(self, preds):
        return filter_person(super().postprocess(preds))


def filter_person(predictions):
    return [{key: value[pred["cls"] == 0] for key, value in pred.items()} for pred in predictions]


def match_counts(predictions, truth, threshold=0.5):
    """Confidence-ordered greedy, one-to-one IoU matching (one person class)."""
    matched = set()
    tp = 0
    for pred in sorted(predictions, key=lambda p: p[4], reverse=True):
        best_iou, best = -1.0, None
        for index, box in enumerate(truth):
            if index in matched:
                continue
            overlap = max(0, min(pred[2], box[2]) - max(pred[0], box[0])) * max(
                0, min(pred[3], box[3]) - max(pred[1], box[1])
            )
            union = (
                (pred[2] - pred[0]) * (pred[3] - pred[1])
                + (box[2] - box[0]) * (box[3] - box[1])
                - overlap
            )
            iou = overlap / union if union > 0 else 0
            if iou > best_iou:
                best_iou, best = iou, index
        if best is not None and best_iou >= threshold:
            matched.add(best)
            tp += 1
    return {"tp": tp, "fp": len(predictions) - tp, "fn": len(truth) - tp}


def ground_truth(row):
    w, h = row["width"], row["height"]
    return [
        [(x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h]
        for x, y, bw, bh in row["boxes"]
    ]


def draw_panel(frame, predictions, truth, title):
    frame = frame.copy()
    for box in truth:
        x1, y1, x2, y2 = [round(v) for v in box]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
    for *box, conf in predictions:
        x1, y1, x2, y2 = [round(v) for v in box]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 120, 255), 2)
        cv2.putText(frame, f"{conf:.2f}", (x1, max(18, y1)), 0, 0.5, (0, 120, 255), 1)
    frame = cv2.resize(frame, (640, 360))
    cv2.rectangle(frame, (0, 0), (640, 29), (0, 0, 0), -1)
    cv2.putText(
        frame,
        title + " | green=truth orange=prediction",
        (8, 20),
        0,
        0.42,
        (255, 255, 255),
        1,
    )
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "yolo26n.pt")
    parser.add_argument("--data", type=Path, default=ROOT / "data/fine-tuning/data.yaml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare", type=Path, help="Baseline evaluation directory")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    manifest, manifest_hash = checked_manifest(args.data)
    model_hash = sha256(args.model)
    settings = {
        "imgsz": args.imgsz,
        "device": device_name(args.device),
        "batch": 1,
        "workers": 0,
        "rect": False,
        "conf": 0.001,
        "iou": 0.7,
        "max_det": 300,
        "classes": [0],
        "augment": False,
        "seed": 42,
        "deterministic": True,
        "quantize": 32,
    }
    baseline = None
    if args.compare:
        baseline = json.loads((args.compare / "summary.json").read_text())
        if baseline["manifest_sha256"] != manifest_hash or baseline["settings"] != settings:
            raise ValueError("Baseline must use the same split and evaluation settings")
    output = fresh_output(args.output)
    model = YOLO(str(args.model))
    if model.names.get(0) != "person":
        raise ValueError("Expected person at class 0")
    started = time.perf_counter()
    metrics = model.val(
        data=str(args.data.resolve()),
        validator=PersonValidator,
        project=str(output),
        name="validation",
        plots=False,
        verbose=False,
        **settings,
    )
    # Predict on loaded images: timing excludes decoding, includes preprocess/model/postprocess.
    predict_settings = {
        k: v for k, v in settings.items() if k not in {"batch", "workers", "seed", "deterministic"}
    }
    predict_settings["conf"] = 0.25
    val_rows = [row for row in manifest["rows"] if row["split"] == "val"]
    first = cv2.imread(str(args.data.parent / "images/val" / val_rows[0]["filename"]))
    for _ in range(5):
        model.predict(first, verbose=False, **predict_settings)
    records, times = [], []
    totals = {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "empty_images": 0,
        "empty_images_with_fp": 0,
        "empty_frame_fp": 0,
    }
    for row in val_rows:
        frame = cv2.imread(str(args.data.parent / "images/val" / row["filename"]))
        if settings["device"] != "cpu" and torch.cuda.is_available():
            torch.cuda.synchronize()
        before = time.perf_counter()
        result = model.predict(frame, verbose=False, **predict_settings)[0]
        if settings["device"] != "cpu" and torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - before
        predictions = np.column_stack(
            (result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy())
        ).tolist()
        truth = ground_truth(row)
        counts = match_counts(predictions, truth)
        times.append(elapsed)
        records.append(
            {
                "filename": row["filename"],
                "day": row["day"],
                "predictions": predictions,
                "counts": counts,
                "latency_seconds": elapsed,
            }
        )
        for key in counts:
            totals[key] += counts[key]
        if not truth:
            totals["empty_images"] += 1
            totals["empty_images_with_fp"] += counts["fp"] > 0
            totals["empty_frame_fp"] += counts["fp"]
        if row["filename"] in manifest["config"]["comparison_frames"]:
            panel = draw_panel(frame, predictions, truth, output.name)
            visual_dir = output / "visuals"
            visual_dir.mkdir(exist_ok=True)
            cv2.imwrite(str(visual_dir / row["filename"]), panel)
            if baseline:
                previous = cv2.imread(str(args.compare / "visuals" / row["filename"]))
                if previous is None:
                    raise ValueError("Baseline comparison image missing")
                cv2.imwrite(
                    str(visual_dir / ("comparison_" + row["filename"])),
                    np.hstack([previous, panel]),
                )
    tp, fp, fn = (totals[k] for k in ("tp", "fp", "fn"))
    summary = {
        "model": args.model.name,
        "model_sha256": model_hash,
        "manifest_sha256": manifest_hash,
        "settings": settings,
        "environment": environment(),
        "metrics": {k: float(v) for k, v in metrics.results_dict.items()},
        "fixed_threshold": {
            "confidence": 0.25,
            "match_iou": 0.5,
            **totals,
            "precision": tp / (tp + fp) if tp + fp else 0,
            "recall": tp / (tp + fn) if tp + fn else 0,
        },
        "timing": {
            "method": "5 warmups then batch-1 loaded-image predict; includes preprocessing and postprocessing; excludes decoding",
            "mean_ms": float(np.mean(times) * 1000),
            "median_ms": float(np.median(times) * 1000),
            "p95_ms": float(np.percentile(times, 95) * 1000),
            "fps": float(1 / np.mean(times)),
            "evaluation_seconds": time.perf_counter() - started,
        },
    }
    write_json(output / "predictions.json", records)
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
