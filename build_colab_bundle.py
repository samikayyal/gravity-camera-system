"""Package the frozen prepared dataset and generate its hash-pinned Colab cell."""

import json
import re
import subprocess
from zipfile import ZIP_DEFLATED, ZipFile

from prepare_dataset import ROOT, materialize, sha256, write_json


def main():
    manifest = json.loads((ROOT / "data/fine-tuning/manifest.json").read_text())
    materialize(manifest, ROOT / "data", ROOT / "data/fine-tuning")
    files = [
        ROOT / name
        for name in [
            "prepare_dataset.py",
            "experiment.py",
            "train.py",
            "evaluate.py",
            "colab_experiment.py",
            "yolo26n.pt",
        ]
    ]
    files += [
        p for p in (ROOT / "data/fine-tuning").rglob("*") if p.is_file() and p.suffix != ".cache"
    ]
    provenance = {
        "base_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "files": {p.relative_to(ROOT).as_posix(): sha256(p) for p in files},
    }
    write_json(ROOT / "runs/bundle-provenance.json", provenance)
    bundle = ROOT / "runs/gravity-colab-input.zip"
    with ZipFile(bundle, "w", ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(ROOT).as_posix())
        archive.write(ROOT / "runs/bundle-provenance.json", "bundle-provenance.json")
    digest = sha256(bundle)
    template = ROOT / "experiments/camera1/colab_cell.py"
    cell = re.sub(r'expected = "[a-f0-9]{64}"', f'expected = "{digest}"', template.read_text())
    template.write_text(cell, encoding="utf-8")
    write_json(
        ROOT / "experiments/camera1/colab-bundle.json",
        {
            "sha256": digest,
            "bytes": bundle.stat().st_size,
            "provenance": provenance,
        },
    )
    print(f"Upload: {bundle}\nSHA256: {digest}\nCell: {template}")


if __name__ == "__main__":
    main()
