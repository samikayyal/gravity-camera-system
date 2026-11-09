import copy
import io
from pathlib import Path
from zipfile import ZipFile

import pytest
import torch

from evaluate import filter_person, match_counts
from prepare_dataset import assign_split, indexed_members, materialize, parse_labels


@pytest.mark.parametrize(
    "text",
    [
        "1 .5 .5 .1 .1",
        "0 nan .5 .1 .1",
        "0 .9 .5 .4 .1",
        "0 .5 .5 0 .1",
        "0 .5 .5 -.1 .1",
        "0 .5 .5 .1",
    ],
)
def test_rejects_invalid_labels(text):
    with pytest.raises(ValueError):
        parse_labels(text, "fixture")


def test_explicit_empty_and_edge_boxes():
    assert parse_labels(" \n", "negative") == []
    assert parse_labels("0 .5 .5 1 1\n", "edge") == [[0.5, 0.5, 1.0, 1.0]]


def test_zip_rejects_duplicate_basenames_and_traversal():
    for members in [["one/a.jpg", "two/a.jpg"], ["../a.jpg"]]:
        stream = io.BytesIO()
        with ZipFile(stream, "w") as archive:
            for name in members:
                archive.writestr(name, b"image")
        with ZipFile(stream) as archive, pytest.raises(ValueError):
            indexed_members(archive, {".jpg"})


def test_split_keeps_blocks_and_gap_and_represents_both_days():
    rows = [
        {
            "day": day,
            "timestamp_seconds": second,
            "boxes": [] if second % 10 == 0 else [[0.5, 0.5, 0.1, 0.1]],
        }
        for day in ["day1", "day2"]
        for second in range(0, 300, 5)
    ]
    config = {
        "block_seconds": 60,
        "gap_seconds": 30,
        "validation_blocks": {"day1": [2], "day2": [1]},
    }
    original = copy.deepcopy(rows)
    result = assign_split(rows, config)
    assert rows == original
    for day in ["day1", "day2"]:
        selected = [r for r in result if r["day"] == day]
        val = [r for r in selected if r["split"] == "val"]
        train = [r for r in selected if r["split"] == "train"]
        assert len(val) == 12
        assert all(
            abs(v["timestamp_seconds"] - t["timestamp_seconds"]) >= 30 for v in val for t in train
        )
        assert {r["block"] for r in val}.isdisjoint(r["block"] for r in train)
        assert sum(r["split"] == "excluded" for r in selected) == 12


def test_invalid_split_does_not_silently_proceed():
    rows = [{"day": "day1", "timestamp_seconds": 5, "boxes": [[0.5, 0.5, 0.1, 0.1]]}]
    with pytest.raises(ValueError):
        assign_split(
            rows,
            {
                "block_seconds": 60,
                "gap_seconds": 30,
                "validation_blocks": {"day1": [12], "day2": [1]},
            },
        )


def test_materialize_refuses_to_overwrite_existing_directory(tmp_path):
    sentinel = tmp_path / "keep.txt"
    sentinel.write_text("keep")
    with pytest.raises(ValueError, match="already exists"):
        materialize({"rows": []}, Path("unused"), tmp_path)
    assert sentinel.read_text() == "keep"


def test_person_filter_does_not_relabel_other_coco_classes():
    predictions = [
        {
            "cls": torch.tensor([0.0, 1.0, 2.0]),
            "conf": torch.tensor([0.3, 0.8, 0.9]),
            "bboxes": torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]),
        }
    ]
    result = filter_person(predictions)[0]
    assert result["cls"].tolist() == [0.0]
    assert result["bboxes"].tolist() == [[1, 2, 3, 4]]


def test_matching_counts_duplicate_prediction_and_missing_person():
    truth = [[0, 0, 10, 10], [20, 20, 30, 30]]
    predictions = [[0, 0, 10, 10, 0.9], [0, 0, 10, 10, 0.8], [50, 50, 60, 60, 0.7]]
    assert match_counts(predictions, truth) == {"tp": 1, "fp": 2, "fn": 1}
    assert match_counts(predictions, []) == {"tp": 0, "fp": 3, "fn": 0}
    assert match_counts([], truth) == {"tp": 0, "fp": 0, "fn": 2}


def test_matching_requires_iou_threshold():
    assert match_counts([[0, 0, 5, 5, 0.9]], [[0, 0, 10, 10]]) == {
        "tp": 0,
        "fp": 1,
        "fn": 1,
    }
