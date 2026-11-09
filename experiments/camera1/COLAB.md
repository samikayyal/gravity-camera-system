# Colab follow-up: prepared, awaiting GPU capacity

On 2026-11-09 the owner authorized running the 20-epoch follow-up in the open
Chrome Colab notebook, including uploading the required data. The empty Fine
Tuning code cell was filled and Colab confirmed **All changes saved**. Existing
installation, Drive, prediction, and extraction cells were preserved.

Two T4 allocation attempts returned: **Failed to assign a backend — No backend
with GPU available.** T4 was the only enabled GPU option; premium GPUs were
disabled. No purchase was made. No dataset upload, dependency installation, or
training occurred in Colab. Issue [#2](https://github.com/samikayyal/gravity-camera-system/issues/2)
remains open. The earlier CPU smoke results are unchanged.

## Prepared files

- `build_colab_bundle.py`: verifies the existing prepared dataset, packages only
  the retained train/validation images and labels, frozen manifest, original
  weights, and required scripts. It records hashes of every input and updates
  the SHA-256 expected by the cell template.
- `experiments/camera1/colab_cell.py`: reusable notebook cell. Run this cell only,
  with a GPU runtime; do not Run all because the notebook contains unrelated
  older cells. The saved notebook has equivalent code and the same bundle hash.
- `colab_experiment.py`: rewrites only the machine-specific dataset YAML path,
  verifies the dataset through the existing workflow, evaluates the pretrained
  model, trains for 20 epochs, evaluates `last.pt`, and packages results.
- `experiments/camera1/colab-bundle.json`: exact upload size/hash and file hashes.
- Private local upload: `runs/gravity-colab-input.zip` (approximately 77 MB).

Rebuild with `uv run build_colab_bundle.py`, then paste the updated cell template
into Colab. Rebuilding can change the archive hash, so use the matching cell.
Original videos and unused excluded frames are not uploaded. The bundle is
ignored by Git because it contains private camera images and model weights.

## Execution when a T4 becomes available

1. Connect to the T4 runtime and run the prepared Fine Tuning cell.
2. Select the local `runs/gravity-colab-input.zip` when the upload prompt appears.
3. The cell verifies the ZIP and all bundled files, installs `uv` from its
   official installer, creates a virtual environment using Colab's installed
   GPU-compatible PyTorch, and installs `ultralytics==8.4.204`, PyYAML, and psutil
   through `uv`. This setup has not yet been exercised on Colab.
4. It runs the baseline, 20-epoch training, and final comparison sequentially.
   Settings retain 640 pixels, batch 2, FP32, AdamW, seed 42, workers 0, and the
   frozen dataset. Twenty epochs start from the original pretrained weights.
5. It downloads `gravity-colab20-results.zip`, including logs, both checkpoints,
   numeric metrics, environment versions, and private visual comparisons.

The cell requires a fresh `/content/gravity-camera1` directory and rejects
accidental reruns into existing output. After a partial failure, preserve useful
outputs and use a fresh runtime or explicitly chosen new work directory; do not
blindly rerun. Logs are under `/content/gravity-camera1/runs/colab20/` while the
experiment is running.

The Colab environment intentionally retains its CUDA-compatible torch build;
it is not the Windows lock environment. All installed versions and GPU details
are captured. Compare baseline and fine-tuned results within that same runtime.
GPU throughput is not directly comparable to the prior CPU measurements.

After downloading, retain the result archive/checkpoints/images privately under
`runs/`, and commit only code, provenance, numeric results, and a findings report.
No Colab performance or training success is claimed until execution completes.

Validation so far: scripts compile, Ruff checks pass, and the upload's archived
files match their recorded hashes. Runtime, dependency installation, and model
execution remain unverified because GPU allocation failed.
