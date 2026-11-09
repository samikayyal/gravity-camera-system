import hashlib
from zipfile import ZipFile

import pytest

from prepare_dataset import SOURCES, materialize


def fixture_dataset(tmp_path):
    data = tmp_path / "data"
    (data / "annotations").mkdir(parents=True)
    rows = []
    for day, (image_zip, _, _, prefix, _) in SOURCES.items():
        filename = prefix + "0000.jpg"
        content = day.encode()
        with ZipFile(data / "annotations" / image_zip, "w") as archive:
            archive.writestr("nested/" + filename, content)
        rows.append(
            {
                "day": day,
                "filename": filename,
                "split": "train" if day == "day1" else "val",
                "image_sha256": hashlib.sha256(content).hexdigest(),
                "boxes": [[0.5, 0.5, 0.1, 0.1]] if day == "day1" else [],
            }
        )
    manifest = {"rows": rows}
    output = tmp_path / "generated"
    materialize(manifest, data, output)
    return manifest, data, output


def test_identical_rerun_and_negative_label(tmp_path):
    manifest, data, output = fixture_dataset(tmp_path)
    before = {str(p): p.read_bytes() for p in output.rglob("*") if p.is_file()}
    materialize(manifest, data, output)
    assert before == {str(p): p.read_bytes() for p in output.rglob("*") if p.is_file()}
    assert (output / "labels/val/camera_1_2_0000.txt").read_text() == ""


@pytest.mark.parametrize("corruption", ["image", "label", "yaml", "stale", "missing"])
def test_rerun_rejects_modified_or_stale_data(tmp_path, corruption):
    manifest, data, output = fixture_dataset(tmp_path)
    targets = {
        "image": "images/train/cam01_0000.jpg",
        "label": "labels/train/cam01_0000.txt",
        "yaml": "data.yaml",
        "stale": "images/train/extra.jpg",
        "missing": "labels/val/camera_1_2_0000.txt",
    }
    path = output / targets[corruption]
    if corruption == "missing":
        path.unlink()
    elif corruption == "label":
        path.write_text("0 .4 .4 .1 .1\n")
    else:
        path.write_text("corrupted")
    with pytest.raises(ValueError):
        materialize(manifest, data, output)


def test_generated_caches_are_allowed(tmp_path):
    manifest, data, output = fixture_dataset(tmp_path)
    (output / "labels/train.cache").write_bytes(b"cache")
    materialize(manifest, data, output)
