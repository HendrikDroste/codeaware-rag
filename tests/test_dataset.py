"""Check pinned questions, source boundaries and dataset integrity."""

import csv
import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.dataset import (
    METRIC_VERSION,
    extract_file_path_from_url,
    load_benchmark,
    normalize_source_path,
)
from src.index_state import file_hashes


def write_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def benchmark(tmp_path):
    root = tmp_path / "repo"
    source = root / "src/flask/a.py"
    source.parent.mkdir(parents=True)
    source.write_text("x = 1\n" * 10)
    (source.parent / "cli.py").write_text("pass\n")
    directory = tmp_path / "benchmark"
    directory.mkdir()
    revision = "a" * 40
    url = f"https://github.com/example/repo/blob/{revision}/src/flask/a.py"
    write_rows(
        directory / "validation.csv",
        [
            {
                "Question": "Find a",
                "URL": url,
                "start_line": "1",
                "end_line": "10",
            }
        ],
    )
    manifest = {
        "schema_version": 1,
        "dataset_id": "fixture-v2",
        "repository_url": "https://github.com/example/repo",
        "revision": revision,
        "path_format": "repository-relative",
        "metric_version": METRIC_VERSION,
        "question_count": 1,
        "validation_sha256": hashlib.sha256(
            (directory / "validation.csv").read_bytes()
        ).hexdigest(),
        "file_hashes": file_hashes(root),
    }
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return root, directory, manifest


def test_paths_are_repository_relative_and_keep_src(benchmark):
    root, _, manifest = benchmark
    url = (
        f"{manifest['repository_url']}/blob/{manifest['revision']}/"
        "src/flask/a.py#L1"
    )
    assert extract_file_path_from_url(url) == "src/flask/a.py"
    for source in (
        "src/flask/a.py",
        "src\\flask\\a.py",
        "./src/flask/a.py",
        str(root / "src/flask/a.py"),
    ):
        assert normalize_source_path(source, root) == "src/flask/a.py"
    for source in (
        "../outside.py",
        str(root.parent / "outside.py"),
        "missing.py",
        "flask/a.py",
    ):
        with pytest.raises(ValueError):
            normalize_source_path(source, root)


@pytest.mark.parametrize(
    "change",
    ["csv", "edit", "add", "delete", "revision", "range", "counts", "metric"],
)
def test_pinned_benchmark_rejects_changed_inputs(benchmark, change):
    root, directory, manifest = benchmark
    path = directory / "validation.csv"
    if change == "csv":
        path.write_text(path.read_text() + "\n")
    elif change == "edit":
        (root / "src/flask/a.py").write_text("pass\n" * 10)
    elif change == "add":
        (root / "extra.py").write_text("pass\n")
    elif change == "delete":
        (root / "src/flask/a.py").unlink()
    elif change in {"revision", "range"}:
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        if change == "revision":
            rows[0]["URL"] = rows[0]["URL"].replace("a" * 40, "b" * 40)
        else:
            rows[0]["end_line"] = "11"
        write_rows(path, rows)
        manifest["validation_sha256"] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    elif change == "counts":
        manifest["question_count"] = 2
    else:
        manifest["metric_version"] = "old"
    (directory / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        load_benchmark(directory, root)


def test_loader_returns_questions_and_provenance(benchmark):
    root, directory, manifest = benchmark
    data = load_benchmark(directory, root)
    assert data["questions"] == ["Find a"]
    assert data["metadata"] == {
        key: manifest[key]
        for key in ("dataset_id", "revision", "metric_version")
    }


@pytest.mark.parametrize("change", ["missing_url", "mismatch", "negative"])
def test_malformed_labels_are_rejected(benchmark, change):
    root, directory, manifest = benchmark
    path = directory / "validation.csv"
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if change == "missing_url":
        rows[0]["URL"] = ""
    elif change == "mismatch":
        rows[0]["URL"] += ";" + rows[0]["URL"]
    else:
        rows[0]["start_line"] = "0"
    write_rows(path, rows)
    manifest["validation_sha256"] = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()
    (directory / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        load_benchmark(directory, root)


def test_committed_dataset_preserves_questions_and_pins_one_revision():
    directory = Path(__file__).resolve().parents[1] / "benchmarks/flask"
    manifest = json.loads((directory / "manifest.json").read_text())
    original = directory / "original-validation.csv"
    updated = directory / "validation.csv"
    assert (
        hashlib.sha256(original.read_bytes()).hexdigest()
        == manifest["historical_validation_sha256"]
    )
    assert (
        hashlib.sha256(updated.read_bytes()).hexdigest()
        == manifest["validation_sha256"]
    )
    with original.open(newline="") as stream:
        old_rows = list(csv.DictReader(stream))
    with updated.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == manifest["question_count"] == 18
    assert [row["Question"] for row in rows] == [
        row["Question"] for row in old_rows
    ]
    for row in rows:
        urls = row["URL"].split(";")
        starts, ends = row["start_line"].split(";"), row["end_line"].split(";")
        assert len(urls) == len(starts) == len(ends)
        for url, start, end in zip(urls, starts, ends):
            assert f"/blob/{manifest['revision']}/" in url
            assert extract_file_path_from_url(url) in manifest["file_hashes"]
            assert 1 <= int(start) <= int(end)
