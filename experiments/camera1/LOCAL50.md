# Camera 1: 50-epoch local follow-up

On 2026-11-09 the owner changed the follow-up length to **50 epochs** and requested
local execution after Colab could not allocate a GPU. This supersedes the pending
20-epoch Colab run. Tracking: [issue #2](https://github.com/samikayyal/gravity-camera-system/issues/2).

## Completed results — reviewed 2026-11-10

All 50 epochs and both standalone evaluations completed. Training took
**10,966.92 seconds (3 hours 2 minutes 47 seconds)**, including the trainer's final
validation. Peak process working set was 1,099,481,088 bytes (about 1.02 GiB).
The final checkpoint improves validation mAP50 to **88.21%**, from 45.41% after
three epochs and 8.85% for the pretrained model. No deployment or analytics
integration was performed.

| Metric | Pretrained | 3 epochs | 50 epochs, final checkpoint |
|---|---:|---:|---:|
| mAP50 | 8.85% | 45.41% | **88.21%** |
| mAP50–95 | 5.15% | 25.13% | **51.67%** |
| Precision at confidence 0.25 | 20.69% | 58.72% | **65.79%** |
| Recall at confidence 0.25 | 4.26% | 45.39% | **88.65%** |
| True positives at confidence 0.25 | 6 | 64 | **125** |
| False positives at confidence 0.25 | 23 | 45 | **65** |
| Misses at confidence 0.25 | 135 | 77 | **16** |
| Empty images with false positives | 3/16 | 0/16 | **0/16** |

Fixed-threshold counts use the same confidence-ordered, one-to-one IoU 0.50
matching as the original experiment. There are 141 annotated people across
the 51 validation images. Ultralytics' F1-selected summary precision/recall are
84.74%/81.56%; these use a different operating point and must not be presented
as the fixed-confidence 0.25 result.

The same-session baseline repeated after training exactly reproduces the earlier
baseline accuracy and counts. Both new evaluations have identical settings and
manifest hashes. Mean loaded-image prediction time was 175.51 ms (5.70 FPS) for
the baseline and 165.02 ms (6.06 FPS) for the final checkpoint; medians were
170.07/162.88 ms and p95 values 208.99/183.27 ms. Timing excludes disk decoding,
model loading, and rendering, and does not establish live multi-camera throughput.
The small timing difference from single sequential runs is not a speed guarantee.

### Remaining errors and a confidence trade-off

The 50-epoch model's Day 1 TP/FP/FN counts are **37/11/7** (44 annotations), and
Day 2 counts are **88/54/9** (97 annotations). All 65 unmatched detections occur
in images containing people. They are not necessarily 65 background objects:
duplicate and poorly localized boxes also count as false positives.

All nine preselected comparison images were inspected. They show substantially
better foreground and distant-person detection, and removal of the baseline's
large padding false alarm. Overlapping boxes around the same people remain
visible, especially around distant Day 2 people, and some small/occluded people
are still missed. Reviewed annotations were preserved. Zero false positives on
16 empty validation frames is encouraging but limited evidence.

An exploratory review filtered the already saved predictions at higher
confidence thresholds, without new inference or changing the original result:

| Confidence | TP | FP | FN | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| 0.25 | 125 | 65 | 16 | 65.79% | 88.65% |
| **0.40** | **119** | **24** | **22** | **83.22%** | **84.40%** |
| 0.50 | 106 | 16 | 35 | 86.89% | 75.18% |
| 0.60 | 84 | 9 | 57 | 90.32% | 59.57% |
| 0.70 | 65 | 2 | 76 | 97.01% | 46.10% |

Confidence 0.40 is a useful candidate for a subsequent video trial: it removes
41 unmatched detections at the cost of six additional misses relative to 0.25.
This is post-hoc development analysis on the validation set, not independent
test evidence or a changed production default. Threshold choice should reflect
the acceptable false-occupancy versus missed-person trade-off.

### Interpretation and next step

The trainer's highest mAP50–95 occurred at epoch 41 (0.52194); the last five
epochs range from 0.50800 to 0.52066, with epoch 50 at 0.51633. That suggests
gains were leveling off in this run; it does not prove the optimal training
length or establish overfitting. Trainer validation differs slightly from the
common standalone evaluation, so the headline table uses only standalone
results. `best.pt` is preserved, but the reported comparison remains `last.pt`
as planned; the best checkpoint was not separately evaluated here.

The next useful experiment is a fresh Camera 1 recording with occupied and empty
periods, small people, occlusion, and varied activity/participants. Freeze the
checkpoint and candidate settings before evaluating that recording. Inspect
video continuity and overlapping detections before integrating tracking or
occupancy. Additional epochs alone are not the first recommendation from this
result. All current scores remain development validation on two short,
correlated recordings from the same viewpoint.

## Configuration and provenance

Settings: original pretrained YOLO26n, 50 epochs, 640-pixel input, batch 2,
workers 0, seed 42, FP32, AdamW at initial LR 0.001 with final factor 0.01,
no warmup, no early stopping, and mosaic enabled throughout. Fifty epochs start
from the original weights, not from the three-epoch checkpoint. The longer run
also stretches the same learning-rate schedule over fifty epochs.

The frozen dataset remains 207 training images / 51 validation images / 51
boundary-excluded images. The launch verified these hashes:

- Manifest: `250bb16c086dbde0b80ebca90b6436e32b853ebd43f5c6da8a92fb4a5a7c54aa`.
- Pretrained weights: `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`.

The existing Windows `uv` environment is used, with Ultralytics 8.4.204 and
PyTorch 2.15.0 CPU. No packages or training implementation were changed for this
run. Ultralytics selected seven CPU threads. Actual runtime was shorter than
the initial 3.5–5-hour estimate.

## Commands and artifacts

The following stages were launched as a sequential pipeline. Each subsequent
stage executes only if the previous command succeeds:

```powershell
uv run train.py --epochs 50 --device cpu --output runs/camera1-local50
uv run evaluate.py --device cpu --output runs/camera1-local50-baseline
uv run evaluate.py --device cpu --model runs/camera1-local50/training/weights/last.pt --output runs/camera1-local50-evaluation --compare runs/camera1-local50-baseline
```

Private local outputs:

- Training log: `runs/local50-training.log`.
- Per-epoch metrics: `runs/camera1-local50/training/results.csv`.
- Checkpoints: `runs/camera1-local50/training/weights/last.pt` and `best.pt`.
- Training completion summary: `runs/camera1-local50/summary.json`.
- Baseline log/results: `runs/local50-baseline.log` and `runs/camera1-local50-baseline/`.
- Final evaluation log/results: `runs/local50-evaluation.log` and `runs/camera1-local50-evaluation/`.

The final evaluation uses `last.pt`, as in the smoke comparison. All nine
preselected side-by-side comparisons exist under its `visuals/` directory.

Committed numeric artifacts are in [results-local50/](results-local50/): baseline
and final evaluation summaries, per-image predictions and counts, all 50 epochs,
full resolved training arguments with repository-relative paths, training
runtime/environment/hash summary, and the exploratory threshold review.
Private images, checkpoint binaries, and full logs remain local and ignored.

Verified checkpoint SHA-256 hashes:

- `last.pt`: `6004036105fce83d9252e07d3053b8744d7f25fd85b16dd97550fa08071b46a3`.
- `best.pt`: `2324faa4d91124bdfcb3603aa4fdb71966a13f18d05fa78dc9d66d0944992108`.

Completion verification: exactly 50 sequential epoch rows; training and both
evaluation summaries present; checkpoint, initial-weight, and frozen-manifest
hashes agree; baseline/final evaluation settings match; the dataset passes the
existing rerun integrity check; all nine comparison images decode successfully
and were visually inspected. No training or inference code changed during this
closeout, and no additional training was launched.
