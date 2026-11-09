# Camera 1 detection experiment — 2026-11-09

The three-epoch CPU smoke test completed successfully. Fine-tuning improved
validation mAP50 from **8.85% to 45.41%**, but the final checkpoint still missed
77 of 141 annotated people at confidence 0.25 and produced 45 unmatched
detections. This is encouraging development evidence, not a production-ready
detector or an independent test result. No longer training run was launched.

Specification and completion tracking: [issue #1](https://github.com/samikayyal/gravity-camera-system/issues/1).

## Sources and frozen split

- Day 1: 59 images, 166 person boxes; video duration 298.600 seconds, 25 FPS.
- Day 2: 250 images, 324 person boxes; video duration 761.962 seconds,
  approximately 25 FPS. The owner's corrected export is `camera_1_2_1.zip`.
- The CVAT image inventories match all 309 extracted images exactly. Day 2
  omitted 165 negative label files; these were converted to explicit empty
  labels using its complete image inventory. Reviewed boxes were not changed.
- All 309 images decoded, with no duplicate decoded pixel content. All matched
  their reconstructed extraction points in the corresponding source videos.
  Maximum resized-image mean absolute channel difference was 0.9427 on a
  0–255 scale, below the predeclared 2.0 audit threshold. This supports the
  renamed Day 2 numbering and finds no stale Day 1 frames. Timestamps remain
  reconstructed seek requests, not independently saved extraction timestamps.
- Source hashes, annotations, frame indices, reconstructed times, and every
  assignment are in [manifest.json](manifest.json).

Validation uses Day 1 seconds [120, 180) and Day 2 [60, 120) and [600, 660).
These complete 60-second blocks were selected using annotation counts and
source-image review before model inference. The initially proposed 15-second
gap was increased to 30 seconds because activity continued across boundaries.
Training samples within 30 seconds of the validation intervals are excluded.

| Split | Day 1 images | Day 2 images | Total images | Person boxes | Empty images |
|---|---:|---:|---:|---:|---:|
| Train | 36 | 171 | 207 | 204 | 137 |
| Validation | 12 | 39 | 51 | 141 | 16 |
| Excluded | 11 | 40 | 51 | 145 | 12 |

Validation is 19.77% of retained images. Negative images dominate training
(137/207); only 70 training images contain people. Blocks and gaps reduce
adjacency but do not eliminate same-session, same-participant correlation.
Nine visual comparison frames were selected in [split.json](split.json) before
evaluation, and the manifest was frozen before the initial baseline.

## Evaluation results

Both models were evaluated separately with square 640-pixel inputs, batch 1,
FP32, CPU, person-only outputs, no test-time augmentation, IoU setting 0.7,
max detections 300, and the same 51 validation images. The explicit validator
filter is necessary because this installed Ultralytics validator does not pass
its `classes` option into postprocessing. Other COCO classes are removed, not
relabelled as people.

| Metric | Pretrained YOLO26n | Final three-epoch checkpoint |
|---|---:|---:|
| Precision, Ultralytics F1-selected point | 20.43% | 58.21% |
| Recall, Ultralytics F1-selected point | 14.18% | 45.39% |
| mAP50 | 8.85% | 45.41% |
| mAP50–95 | 5.15% | 25.13% |
| Precision at confidence 0.25 | 20.69% | 58.72% |
| Recall at confidence 0.25 | 4.26% | 45.39% |
| True positives at confidence 0.25 | 6 | 64 |
| False positives at confidence 0.25 | 23 | 45 |
| Misses at confidence 0.25 | 135 | 77 |
| Empty frames with false positives | 3 / 16 | 0 / 16 |
| Mean loaded-image prediction latency | 189.49 ms | 203.72 ms |
| Median loaded-image prediction latency | 180.78 ms | 180.61 ms |
| p95 loaded-image prediction latency | 231.40 ms | 304.10 ms |
| Loaded-image prediction throughput | 5.28 FPS | 4.91 FPS |

mAP uses confidence floor 0.001. Fixed-threshold counts use confidence-ordered,
one-to-one matching at IoU 0.50. An unmatched detection may be a background
false alarm, duplicate, or poorly localized person. Absolute false positives
increased even though precision improved because the fine-tuned model detects
many more objects. Zero errors on 16 empty images is limited evidence, not a
guarantee about empty-scene behavior.

At the fixed threshold, Day 1 changes from TP/FP/FN **2/2/42** to **32/16/12**;
Day 2 changes from **4/21/93** to **32/29/65**. Day 2 remains especially weak.
Inspected comparisons show tighter foreground-person boxes and removal of a
padding false alarm, but also duplicate/misaligned boxes and missed distant
people. The authoritative annotations were preserved.

Timing uses five warmups followed by one prediction per loaded image. It
includes preprocessing and postprocessing but excludes decoding, model loading,
and rendering. The initial baseline ran before training; a second baseline and
the smoke evaluation ran sequentially after training, without the test suite
running concurrently. The two baseline runs have identical accuracy metrics
and fixed-threshold counts. Reported timing uses the second baseline. Similar
medians and differing tails do not establish a meaningful model speed change.
These figures do not represent end-to-end video or multi-camera throughput.

## Training and cost

Training used seed 42, AdamW, initial learning rate 0.001, final LR factor 0.01,
batch 2, 640-pixel inputs, zero DataLoader workers, FP32, three epochs, no warmup,
no early stopping, and mosaic enabled throughout. Ultralytics retained its
nominal batch size 64 and gradient accumulation defaults. This was a short
pipeline test, not a tuned training schedule. The log confirmed all 708 weight
items transferred and the person classification row remapped by class name.

The run took **788.85 seconds (13 minutes 9 seconds)**, including final built-in
validation. Peak process working set was **1,098,100,736 bytes (1.02 GiB)**.
PyTorch reports CPU-only operation; Ultralytics selected seven CPU threads on
this Windows laptop. Runtime budgeting is approximate: a test run and light
diagnostics also ran during part of early training.

The final-epoch `last.pt` was used for the comparison as planned. `best.pt` is
retained locally but was not substituted into this comparison. Per-epoch
validation was not monotonic, and internal trainer validation differs slightly
from the common standalone evaluation settings; the table above uses only the
standalone evaluations.

A concrete next trial would be **20 epochs from the original pretrained
weights on this same frozen split**, initially retaining the recorded settings
to isolate the effect of longer training. Linear extrapolation is about
88 minutes; reserve **90–120 minutes** on this CPU. This is a budget proposal,
not a promise of improvement, and the run has not been started. A GPU environment
is a reasonable alternative if iteration speed becomes limiting. More occupied
and challenging Camera 1 footage, especially the Day 2 failure cases, is also
needed. Keep a new recording untouched for the eventual test assessment.

No tracking, occupancy, dashboard, live-camera, or other-camera integration was
changed. Reliability criteria for downstream occupancy still need to be set.

## Reproduction and artifacts

Use the setup and preparation commands in [README](../../README.md). Exact run
order in this experiment:

```powershell
uv run prepare_dataset.py --audit-only
uv run prepare_dataset.py
uv run evaluate.py --output runs/camera1-baseline
uv run train.py
uv run evaluate.py --output runs/camera1-baseline-clean
uv run evaluate.py --model runs/camera1-smoke/training/weights/last.pt --output runs/camera1-smoke-evaluation --compare runs/camera1-baseline-clean
```

Committed numeric artifacts in [results/](results/):

- `baseline-before-training.json`: original baseline, captured before training.
- `baseline.json` and `baseline-predictions.json`: repeated baseline and
  per-image detections/counts used for the final comparison.
- `smoke-evaluation.json` and `smoke-predictions.json`: comparable final-checkpoint
  evaluation and per-image detections/counts.
- `training-summary.json`: timing, memory, settings, environment, and hashes.
- `training-epochs.csv`: all three epochs' losses, validation, and learning rates.
- `training-arguments.json`: full resolved training configuration, with workspace
  paths converted to repository-relative paths.

Private local artifacts:

- Dataset: `data/fine-tuning/`.
- Checkpoints: `runs/camera1-smoke/training/weights/{last,best}.pt`.
- Nine side-by-side comparisons:
  `runs/camera1-smoke-evaluation/visuals/comparison_*.jpg`.
- Logs: `runs/baseline.log`, `runs/training.log`, `runs/baseline-clean.log`, and
  `runs/smoke-evaluation.log`.

Provenance hashes:

| Artifact | SHA-256 |
|---|---|
| Local frozen manifest | `250bb16c086dbde0b80ebca90b6436e32b853ebd43f5c6da8a92fb4a5a7c54aa` |
| Pretrained weights | `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef` |
| Final checkpoint | `7f8a62469d2c7395df75350862218357f9fdd07adddbd16afe1a25a039266766` |
| Best checkpoint | `659f85b6e6204f9f5eca7b8567a6516fef64c232db2006caf24284253eb872e0` |

Verification: **21 tests passed**; Ruff lint and formatting checks passed;
the existing generated dataset passed rerun verification; both local manifest
copies matched; all nine selected comparison files were generated. Private
videos, images, ZIPs, checkpoints, and rendered comparisons are excluded from Git.
