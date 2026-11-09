# Paste into the existing notebook's empty Fine Tuning cell.
# Select a T4 GPU runtime, then run this cell only (not Run all).
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile

import torch
from google.colab import files

assert torch.cuda.is_available(), "Colab GPU unavailable. Reconnect to a T4 runtime."
print("GPU:", torch.cuda.get_device_name(0))
bundle = Path("/content/gravity-colab-input.zip")
expected = "0c26dbeac06a5d458204cfde2e5d9b28e5e3eb75bb9d2b2cd4282faf4e706360"
if not bundle.exists():
    print("Upload runs/gravity-colab-input.zip from the Gravity project.")
    files.upload()
assert hashlib.sha256(bundle.read_bytes()).hexdigest() == expected, "Input bundle hash mismatch"
work = Path("/content/gravity-camera1")
work.mkdir(exist_ok=False)
with ZipFile(bundle) as archive:
    for name in archive.namelist():
        target = (work / name).resolve()
        assert target.is_relative_to(work.resolve()), "Unsafe archive member"
    archive.extractall(work)
metadata = json.loads((work / "bundle-provenance.json").read_text())
for name, digest in metadata["files"].items():
    assert hashlib.sha256((work / name).read_bytes()).hexdigest() == digest, name

# Bootstrap uv from its official installer, then use uv for Python dependencies/execution.
uv = Path.home() / ".local/bin/uv"
if not uv.exists():
    subprocess.run(["bash", "-c", "curl -LsSf https://astral.sh/uv/install.sh | sh"], check=True)
venv = work / ".venv-colab"
subprocess.run(
    [str(uv), "venv", "--system-site-packages", "--python", sys.executable, str(venv)], check=True
)
python = venv / "bin/python"
subprocess.run(
    [
        str(uv),
        "pip",
        "install",
        "--python",
        str(python),
        "ultralytics==8.4.204",
        "psutil",
        "pyyaml",
    ],
    check=True,
)
with (work / "colab-environment.txt").open("w") as out:
    subprocess.run([str(uv), "pip", "freeze", "--python", str(python)], stdout=out, check=True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
subprocess.run(
    [str(uv), "run", "--no-project", "--python", str(python), str(work / "colab_experiment.py")],
    cwd=work,
    check=True,
)
files.download(str(work / "gravity-colab20-results.zip"))
