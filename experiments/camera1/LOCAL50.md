# Camera 1: 50-epoch local follow-up

On 2026-11-09 the owner changed the follow-up length to **50 epochs** and requested
local execution after Colab could not allocate a GPU. This supersedes the pending
20-epoch Colab run. Tracking: [issue #2](https://github.com/samikayyal/gravity-camera-system/issues/2).

## Launch record

The run was launched and observed completing its first training batches on CPU.
This is a launch record, not a completion or accuracy report. The three-epoch
smoke experiment remains the most recent completed result until this run ends.

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
run. The prior smoke runtime implies roughly 3.5–5 hours, subject to competing
load, power/thermal behavior, and interruption. Keep the computer awake and
powered for the job to progress.

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

The final evaluation uses `last.pt`, as in the smoke comparison. Nine preselected
side-by-side comparisons will be saved under its `visuals/` directory. The full
run is not complete until the evaluation summary exists and the process exits
successfully. Logs and CSV can show interim progress without implying completion.

After completion, inspect errors, checkpoint hashes, accuracy metrics, fixed
confidence FP/FN counts, empty-frame behavior, timing, and visuals. Add numeric
results and findings to GitHub; preserve private images and checkpoints locally.
Independent test footage is still needed before claiming performance on new
recordings or readiness for occupancy integration.
