"""Run the frozen Camera 1 follow-up in Colab and create a downloadable archive."""

import json
import os
import shutil
import subprocess
import sys

import torch
import yaml

from prepare_dataset import ROOT, dataset_yaml, sha256, write_json


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Select a Colab GPU runtime before launching this experiment")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    dataset = ROOT / "data/fine-tuning"
    (dataset / "data.yaml").write_text(yaml.safe_dump(dataset_yaml(dataset)))
    run_root = ROOT / "runs/colab20"
    run_root.mkdir(parents=True, exist_ok=False)
    shutil.copy2(ROOT / "colab-environment.txt", run_root / "colab-environment.txt")
    write_json(
        run_root / "runtime.json",
        {
            "gpu": torch.cuda.get_device_name(0),
            "cuda": torch.version.cuda,
            "bundle_manifest": json.loads((ROOT / "bundle-provenance.json").read_text()),
            "manifest_sha256": sha256(dataset / "manifest.json"),
        },
    )
    commands = [
        ["evaluate.py", "--device", "0", "--output", "runs/colab20/baseline"],
        ["train.py", "--device", "0", "--epochs", "20", "--output", "runs/colab20/training"],
        [
            "evaluate.py",
            "--device",
            "0",
            "--model",
            "runs/colab20/training/training/weights/last.pt",
            "--output",
            "runs/colab20/evaluation",
            "--compare",
            "runs/colab20/baseline",
        ],
    ]
    for stage, command in zip(["baseline", "training", "evaluation"], commands):
        print(f"GRAVITY_STAGE: {stage}", flush=True)
        with (run_root / f"{stage}.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(
                [sys.executable, *command], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT
            )
        if result.returncode:
            print((run_root / f"{stage}.log").read_text()[-12000:], flush=True)
            raise RuntimeError(f"{stage} failed with code {result.returncode}")
        summary_path = run_root / stage / "summary.json"
        print(summary_path.read_text(), flush=True)
    archive = shutil.make_archive(str(ROOT / "gravity-colab20-results"), "zip", run_root)
    print(f"GRAVITY_COMPLETE: {archive}", flush=True)


if __name__ == "__main__":
    main()
