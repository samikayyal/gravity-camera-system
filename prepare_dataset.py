"""Validate Camera 1 sources, reconstruct sampling, and materialize a frozen split."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent
SOURCES = {
    "day1": ("camera_1_1_frames.zip", "camera_1_1.zip", "camera_1_1.mp4", "cam01_", 59),
    "day2": (
        "camera_1_2_1_frames.zip",
        "camera_1_2_1.zip",
        "camera_1_2.mp4",
        "camera_1_2_",
        250,
    ),
}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def parse_labels(text, source):
    boxes = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5 or parts[0] != "0":
            raise ValueError(f"{source}: expected person class 0 and four coordinates")
        box = np.asarray([float(v) for v in parts[1:]], dtype=float)
        x, y, w, h = box
        if (
            not np.isfinite(box).all()
            or min(w, h) <= 0
            or max(w, h) > 1
            or min(x - w / 2, y - h / 2) < -1e-6
            or max(x + w / 2, y + h / 2) > 1 + 1e-6
        ):
            raise ValueError(f"{source}: invalid normalized bounding box {box}")
        boxes.append(box.tolist())
    return boxes


def indexed_members(archive, suffixes):
    members = {}
    for info in archive.infolist():
        if info.is_dir():
            continue
        path = PurePosixPath(info.filename)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Unsafe ZIP member: {info.filename}")
        if path.suffix.lower() in suffixes:
            if path.name in members:
                raise ValueError(f"Duplicate ZIP filename: {path.name}")
            members[path.name] = info.filename
    return members


def audit_sources(data_dir):
    rows, sources, seen_hashes = [], {}, {}
    for day, (image_zip, label_zip, video_name, prefix, expected) in SOURCES.items():
        image_path = data_dir / "annotations" / image_zip
        label_path = data_dir / "annotations" / label_zip
        video_path = data_dir / video_name
        sources[day] = {p.name: sha256(p) for p in [image_path, label_path, video_path]}
        cap = cv2.VideoCapture(str(video_path))
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            if not cap.isOpened() or fps <= 0 or frame_count <= 0:
                raise ValueError(f"Cannot read valid video metadata: {video_path}")
            duration = frame_count / fps
            count = min(200, max(1, int(duration / 5))) if day == "day1" else 250
            if count != expected:
                raise ValueError(f"{day}: extraction count differs from expected {expected}")
            timestamps = np.linspace(0, max(0, duration - 1), count)
            with ZipFile(image_path) as images, ZipFile(label_path) as labels:
                image_members = indexed_members(images, {".jpg", ".jpeg", ".png"})
                text_members = indexed_members(labels, {".txt"})
                yaml_members = indexed_members(labels, {".yaml"})
                config = yaml.safe_load(labels.read(yaml_members["data.yaml"]))
                if config.get("names") not in ({0: "person"}, ["person"]):
                    raise ValueError(f"{day}: expected a single person class")
                inventory = [
                    PurePosixPath(v.strip()).name
                    for v in labels.read(text_members["train.txt"]).decode("utf-8-sig").splitlines()
                    if v.strip()
                ]
                expected_names = {f"{prefix}{i:04d}.jpg" for i in range(expected)}
                if (
                    set(inventory) != expected_names
                    or len(inventory) != expected
                    or set(image_members) != expected_names
                ):
                    raise ValueError(f"{day}: image inventory mismatch or stale/missing frames")
                label_members = {k: v for k, v in text_members.items() if k != "train.txt"}
                if set(label_members) - {Path(n).stem + ".txt" for n in inventory}:
                    raise ValueError(f"{day}: orphan label files")
                for index, name in enumerate(sorted(image_members)):
                    content = images.read(image_members[name])
                    decoded = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
                    if decoded is None:
                        raise ValueError(f"Unreadable image: {name}")
                    digest = hashlib.sha256(decoded.tobytes()).hexdigest()
                    if digest in seen_hashes:
                        raise ValueError(f"Duplicate decoded image: {name}, {seen_hashes[digest]}")
                    seen_hashes[digest] = name
                    timestamp = float(timestamps[index])
                    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
                    ok, video_frame = cap.read()
                    if not ok or video_frame.shape != decoded.shape:
                        raise ValueError(f"{name}: video comparison failed")
                    mae = float(
                        np.abs(
                            cv2.resize(video_frame, (320, 180)).astype(float)
                            - cv2.resize(decoded, (320, 180))
                        ).mean()
                    )
                    # JPEG compression causes small differences. A mismatch blocks preparation.
                    if mae > 2.0:
                        raise ValueError(
                            f"{name}: reconstructed timestamp is unverified (MAE={mae:.3f})"
                        )
                    label_name = Path(name).stem + ".txt"
                    label_text = (
                        labels.read(label_members[label_name]).decode("utf-8-sig")
                        if label_name in label_members
                        else ""
                    )
                    boxes = parse_labels(label_text, name)
                    rows.append(
                        {
                            "day": day,
                            "filename": name,
                            "index": index,
                            "timestamp_seconds": timestamp,
                            "timestamp_basis": "reconstructed extraction schedule; video image verified",
                            "video_match_mae": mae,
                            "pixel_sha256": digest,
                            "image_sha256": hashlib.sha256(content).hexdigest(),
                            "width": decoded.shape[1],
                            "height": decoded.shape[0],
                            "boxes": boxes,
                            "omitted_negative_label": label_name not in label_members,
                        }
                    )
            sources[day]["video_metadata"] = {
                "fps": fps,
                "frames": int(frame_count),
                "duration_seconds": duration,
            }
        finally:
            cap.release()
    return {"sources": sources, "rows": rows}


def assign_split(rows, config):
    block_seconds, gap = config["block_seconds"], config["gap_seconds"]
    if block_seconds <= 0 or gap < 0:
        raise ValueError("Block length must be positive and gap nonnegative")
    result = []
    for original in rows:
        row = dict(original)
        row["block"] = int(row["timestamp_seconds"] // block_seconds)
        selected = config["validation_blocks"][row["day"]]
        row["split"] = "val" if row["block"] in selected else "train"
        # Remove training frames near validation intervals; retain whole validation blocks.
        if row["split"] == "train" and any(
            block * block_seconds - gap
            <= row["timestamp_seconds"]
            < (block + 1) * block_seconds + gap
            for block in selected
        ):
            row["split"] = "excluded"
        result.append(row)
    for day in SOURCES:
        for split in ("train", "val"):
            subset = [r for r in result if r["day"] == day and r["split"] == split]
            if not subset or not any(r["boxes"] for r in subset):
                raise ValueError(f"{day}/{split} must contain occupied frames")
        available = {r["block"] for r in result if r["day"] == day}
        if set(config["validation_blocks"][day]) - available:
            raise ValueError(f"{day}: nonexistent validation block")
    for split in ("train", "val"):
        if not any(not r["boxes"] for r in result if r["split"] == split):
            raise ValueError(f"{split} must contain empty frames")
    return result


def summarize(rows):
    counts = {}
    for split in ("train", "val", "excluded"):
        subset = [r for r in rows if r["split"] == split]
        counts[split] = {
            "images": len(subset),
            "boxes": sum(len(r["boxes"]) for r in subset),
            "empty_images": sum(not r["boxes"] for r in subset),
            "by_day": dict(Counter(r["day"] for r in subset)),
        }
    return counts


def materialize(manifest, data_dir, output):
    """Write into a new directory; identical reruns verify, never overwrite arbitrary files."""
    marker = output / "manifest.json"
    expected = {"manifest.json", "data.yaml"}
    for row in manifest["rows"]:
        if row["split"] != "excluded":
            expected.add(f"images/{row['split']}/{row['filename']}")
            expected.add(f"labels/{row['split']}/{Path(row['filename']).stem}.txt")
    if output.exists():
        if not marker.exists() or json.loads(marker.read_text(encoding="utf-8")) != manifest:
            raise ValueError(
                "Output already exists with a different/incomplete manifest; use a new --output"
            )
        actual = {p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()}
        # Ultralytics creates disposable label caches on evaluation/training.
        if actual - {"labels/train.cache", "labels/val.cache"} != expected:
            raise ValueError("Output has missing or stale files; use a new --output")
        for row in manifest["rows"]:
            if row["split"] == "excluded":
                continue
            if sha256(output / "images" / row["split"] / row["filename"]) != row["image_sha256"]:
                raise ValueError("Generated image was modified")
            label = output / "labels" / row["split"] / (Path(row["filename"]).stem + ".txt")
            if parse_labels(label.read_text(), str(label)) != row["boxes"]:
                raise ValueError("Generated labels were modified")
        if yaml.safe_load((output / "data.yaml").read_text()) != dataset_yaml(output):
            raise ValueError("Generated data.yaml was modified")
        return
    output.mkdir(parents=True)
    for day, (image_zip, *_) in SOURCES.items():
        with ZipFile(data_dir / "annotations" / image_zip) as archive:
            members = indexed_members(archive, {".jpg", ".jpeg", ".png"})
            for row in manifest["rows"]:
                if row["day"] != day or row["split"] == "excluded":
                    continue
                image = output / "images" / row["split"] / row["filename"]
                label = output / "labels" / row["split"] / (image.stem + ".txt")
                image.parent.mkdir(parents=True, exist_ok=True)
                label.parent.mkdir(parents=True, exist_ok=True)
                image.write_bytes(archive.read(members[row["filename"]]))
                label.write_text(
                    "".join("0 " + " ".join(str(v) for v in box) + "\n" for box in row["boxes"])
                )
    (output / "data.yaml").write_text(yaml.safe_dump(dataset_yaml(output)), encoding="utf-8")
    write_json(marker, manifest)


def dataset_yaml(output):
    return {
        "path": str(output.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "person"},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--config", type=Path, default=ROOT / "experiments/camera1/split.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data/fine-tuning")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    audit = audit_sources(args.data_dir)
    if args.audit_only:
        write_json(ROOT / "runs/source-audit.json", audit)
        blocks = {}
        for row in audit["rows"]:
            key = f"{row['day']}/{int(row['timestamp_seconds'] // 60)}"
            stats = blocks.setdefault(key, {"images": 0, "boxes": 0, "empty": 0})
            stats["images"] += 1
            stats["boxes"] += len(row["boxes"])
            stats["empty"] += not row["boxes"]
        print(json.dumps(blocks, indent=2))
        return
    config = json.loads(args.config.read_text(encoding="utf-8"))
    rows = assign_split(audit["rows"], config)
    manifest = {
        "schema_version": 1,
        "sources": audit["sources"],
        "config": config,
        "summary": summarize(rows),
        "rows": rows,
    }
    frozen = args.config.parent / "manifest.json"
    if frozen.exists() and json.loads(frozen.read_text()) != manifest:
        raise ValueError("Frozen manifest differs; create a new experiment directory")
    materialize(manifest, args.data_dir, args.output)
    write_json(frozen, manifest)
    print(json.dumps(manifest["summary"], indent=2))


if __name__ == "__main__":
    main()
