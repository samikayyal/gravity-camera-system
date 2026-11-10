# Gravity Camera System

Person detection experiments for Gravity trampoline park. The current scope is
fine-tuning pretrained Ultralytics YOLO26n for Camera 1. Tracking, trampoline
occupancy, utilization analytics, dashboards, live CCTV integration, and other
camera viewpoints are later work. A detected person is not a unique visitor.

The experiment specification is tracked in [GitHub issue #1](https://github.com/samikayyal/gravity-camera-system/issues/1).
See [the experiment record](experiments/camera1/REPORT.md) for measured results and limitations.
The [Colab follow-up](experiments/camera1/COLAB.md) was prepared but could not get
GPU capacity. The subsequent [50-epoch local run](experiments/camera1/LOCAL50.md)
completed: validation mAP50 reached **88.21%**. At confidence 0.25, it detected
125 of 141 annotated people, with 65 unmatched detections and 16 misses.
This remains development validation; a new recording is needed for testing.

## Setup

Use Python 3.12 and `uv`. Install the locked environment:

```powershell
uv sync --locked
```

The current Windows environment is CPU-only. Scripts select CUDA when available
and otherwise use CPU; `--device cpu` makes the choice explicit. Ultralytics
selects its CPU thread count internally; the actual count is recorded in results.
`uv.lock` is committed. Add dependencies with `uv add`, and run Python tools
through `uv run`.

## Private source files

Place the following local files under `data/`:

```text
data/
  camera_1_1.mp4
  camera_1_2.mp4
  annotations/
    camera_1_1_frames.zip
    camera_1_1.zip
    camera_1_2_1_frames.zip
    camera_1_2_1.zip
```

`camera_1_2_1.zip` is the corrected Day 2 export, confirmed by the owner despite
the absence of `(2)` in its filename. The checked-in manifest records the exact
hashes. The local pretrained weights are `yolo26n.pt`; scripts require this file
to exist and record its hash rather than silently downloading a replacement.

Source videos, frames, annotation ZIPs, rendered comparisons, and checkpoints
remain local. Git contains code, the lock file, split metadata, numeric results,
and documentation. Do not force-add `data/`, `runs/`, or model weights.

## Prepare the dataset

```powershell
uv run prepare_dataset.py --audit-only
uv run prepare_dataset.py
```

Preparation checks source inventories, labels, image decoding, duplicate decoded
images, and frame chronology against the videos. It reconstructs the extraction
schedule using the original scripts and actual video durations, then compares
every extracted frame with the corresponding decoded video frame. Timestamps
are reconstructed seek requests, not original extraction logs or exact capture
timestamps. Mean absolute RGB difference at 320×180 must be at most 2/255.

Both exports declare one class, `person` (ID 0). Day 2's complete image inventory
lists 250 frames but omits 165 negative label files. Preparation makes those
labels explicitly empty. Missing images, orphan labels, duplicate filenames,
nonfinite coordinates, invalid boxes, and mismatched inventories are errors.

The fixed [split configuration](experiments/camera1/split.json) uses 60-second
blocks with 30-second training exclusions around validation intervals:

| Recording | Validation intervals (seconds from start, end exclusive) |
|---|---|
| Day 1 | 120–180 |
| Day 2 | 60–120 and 600–660 |

The resulting dataset has 207 training images, 51 validation images, and 51
excluded images. Training contains 204 boxes and 137 empty frames; validation
contains 141 boxes and 16 empty frames. Block selection used only source images
and annotations and was frozen before inference. The long same-session activity
means these splits are still correlated. A later recording is needed for testing.

Output is `data/fine-tuning/`, containing matching `images/train`, `images/val`,
`labels/train`, `labels/val`, `data.yaml`, and `manifest.json`. Absolute dataset
paths are generated locally. [The frozen manifest](experiments/camera1/manifest.json)
contains portable source hashes, image assignments, and annotation provenance.

Identical reruns verify existing images, labels, YAML, and file inventory. A
changed source, split, corrupt output, or stale file causes an error. Use a fresh
experiment configuration directory and `--output` for a different experiment;
do not edit the frozen manifest to fit new results. Original data is never removed.

## Baseline, smoke test, and comparison

Run these sequentially to avoid resource contention:

```powershell
uv run evaluate.py --output runs/camera1-baseline
uv run train.py
uv run evaluate.py --model runs/camera1-smoke/training/weights/last.pt --output runs/camera1-smoke-evaluation --compare runs/camera1-baseline
```

Existing run directories are rejected. Use a new `--output` name when repeating
a run. The default training run is exactly three epochs, 640-pixel inputs, batch
size 2, zero DataLoader workers, seed 42, CPU fallback, FP32, and AdamW at initial
learning rate 0.001. Warmup is disabled for this short run, early stopping is
disabled, and mosaic remains enabled throughout. Other augmentation and
accumulation defaults are supplied by the locked Ultralytics version and saved
in `training/args.yaml`. A longer run requires an explicit `--epochs` argument.

Use the final epoch's `last.pt` for the smoke comparison. Ultralytics also saves
and evaluates `best.pt`, but it is not the smoke comparison checkpoint. Three
epochs verify training, checkpoint saving, evaluation, and runtime; they are not
an accuracy target or a production-readiness decision.

Evaluation records:

- Person-only precision, recall, mAP50, and mAP50–95 using Ultralytics validation
  with confidence floor 0.001. Its summary precision/recall use the F1-selected
  point, not the fixed threshold below. An explicit validator filter removes
  non-person COCO outputs without relabeling them as people.
- TP, FP, and FN at confidence 0.25 and IoU 0.50, using confidence-ordered,
  one-to-one matching. Empty-frame false positives are reported separately.
- Batch-1 FP32 prediction latency after five warmups, including preprocessing
  and postprocessing but excluding disk decoding, model loading, and rendering.
  This is detector throughput, not end-to-end video or multi-camera throughput.
- Software versions, hardware information, checkpoint and manifest hashes,
  peak process memory on Windows, numeric predictions, and nine preselected
  visual examples. Green boxes are annotations; orange boxes are predictions.

Comparisons require matching split hashes and evaluation settings. Both models
use square 640-pixel inputs, the same machine, and the same validation frames.
Do not interpret validation checkpoint/threshold selection as untouched test
performance. The future test recording should include new activity/participants,
occupied and empty intervals, small people, and partial occlusion.

## Checks

```powershell
uv run pytest -q
uv run ruff check .
uv run ruff format --check prepare_dataset.py experiment.py evaluate.py train.py tests
```

Tests cover label validity, archive member safety, split separation, stale/corrupt
output rejection, reproducible dataset reruns, explicit negative labels,
person-only filtering, and one-to-one error counting. Tests do not need private
footage or model weights.
