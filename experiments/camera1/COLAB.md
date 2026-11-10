# Colab follow-up: superseded by local training

On 2026-11-09, a 20-epoch Camera 1 follow-up was prepared for Colab. Two T4
allocation attempts failed because no GPU backend was available. No dataset
upload, dependency installation, or training occurred in Colab.

The owner subsequently chose a 50-epoch local run, which completed. See
[LOCAL50.md](LOCAL50.md) for its configuration, results, and limitations.

The unused Colab bundle builder, runner, notebook cell, bundle metadata, and
private upload archive were removed during cleanup. The earlier preparation
is available in Git history at commit `142f65e`. Any future one-off packaging
helpers belong under `.scratch/` as described in the repository instructions.
