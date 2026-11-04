"""Regression checks for pinned labels, path alignment and coverage."""

import csv
import hashlib
import json
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest

from src.index_state import file_hashes
from src.retrievers.metrics import METRIC_VERSION, line_coverage_ratio
from src.retrievers.validate_retriever import (
    extract_file_path_from_url,
    load_benchmark,
    normalize_source_path,
    save_model_results,
    save_results_to_csv,
    validate_retriever,
)


def region(path="src/flask/a.py", start=1, end=10):
    return {"filename": path, "line_start": start, "line_end": end}


@pytest.mark.parametrize(
    "expected,retrieved,score",
    [
        ([region()], [region(), region()], 1.0),
        ([region()], [region(end=6), region(start=4, end=8)], 0.8),
        ([region(end=5), region(start=4)], [region(start=4, end=5)], 0.2),
        ([region(), region("other.py")], [region()], 0.5),
        ([region()], [region("other.py")], 0.0),
        ([], [region()], 0.0),
        ([region()], [], 0.0),
    ],
)
def test_coverage_counts_unique_lines_per_file(expected, retrieved, score):
    assert line_coverage_ratio(expected, retrieved) == pytest.approx(score)


def test_invalid_ranges_do_not_produce_a_score():
    with pytest.raises(ValueError, match="positive and inclusive"):
        line_coverage_ratio([region(start=0)], [region()])


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
    pd.DataFrame(
        [
            {
                "Question": "Find a",
                "URL": url,
                "start_line": "1",
                "end_line": "10",
            }
        ]
    ).to_csv(directory / "validation.csv", index=False)
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
        "flask/a.py",
        "src\\flask\\a.py",
        "./src/flask/a.py",
        str(root / "src/flask/a.py"),
    ):
        assert normalize_source_path(source, root) == "src/flask/a.py"
    for source in (
        "../outside.py",
        str(root.parent / "outside.py"),
        "missing.py",
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
        data = pd.read_csv(path)
        if change == "revision":
            data.loc[0, "URL"] = data.loc[0, "URL"].replace("a" * 40, "b" * 40)
        else:
            data.loc[0, "end_line"] = 11
        data.to_csv(path, index=False)
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


def test_offline_evaluation_normalizes_paths_and_exports_provenance(
    benchmark, tmp_path, monkeypatch
):
    root, directory, manifest = benchmark
    monkeypatch.setattr(
        "requests.get", Mock(side_effect=AssertionError("No downloads"))
    )
    data = load_benchmark(directory, root)
    pipeline = Mock()
    pipeline.invoke.return_value = {
        "documents": [["wrong file", "a", "a again"]],
        "metadatas": [
            [
                region("flask/cli.py", 1, 1),
                region("flask/a.py"),
                region(str(root / "src/flask/a.py")),
            ]
        ],
    }
    results = validate_retriever(pipeline, data)
    assert results["mrr"] == [0.5]
    assert results["line_coverage"] == [1.0]
    assert results["file_match_ratio"] == [1.0]
    assert results["benchmark"]["revision"] == manifest["revision"]
    assert results["expected_metadata"][0][0]["filename"] == "src/flask/a.py"
    output = tmp_path / "retrieval.csv"
    save_results_to_csv(results, str(output))
    saved = pd.read_csv(output)
    assert list(saved["is_match"]) == [False, True, True]
    assert set(saved["metric_version"]) == {METRIC_VERSION}
    assert set(saved["dataset_id"]) == {manifest["dataset_id"]}
    summary = tmp_path / "new-results/model.csv"
    save_model_results(results, "fixture", str(summary))
    assert pd.read_csv(summary).loc[0, "revision"] == manifest["revision"]


def test_malformed_labels_fail_before_querying(benchmark):
    root, directory, _ = benchmark
    data = load_benchmark(directory, root)
    data.loc[0, "URL"] += ";" + data.loc[0, "URL"]
    pipeline = Mock()
    with pytest.raises(ValueError, match="Mismatched label counts"):
        validate_retriever(pipeline, data)
    pipeline.invoke.assert_not_called()


def test_corpus_changes_during_evaluation_are_rejected(benchmark):
    root, directory, _ = benchmark
    data = load_benchmark(directory, root)

    def change_source(query):
        (root / "src/flask/a.py").write_text("pass\n" * 10)
        return {"documents": [[]], "metadatas": [[]]}

    with pytest.raises(ValueError, match="changed after loading"):
        validate_retriever(Mock(invoke=change_source), data)


def test_committed_dataset_preserves_questions_and_pins_one_revision():
    directory = Path(__file__).resolve().parents[1] / "data"
    manifest = json.loads(
        (directory / "benchmarks/flask/manifest.json").read_text()
    )
    original = directory / "validation.csv"
    updated = directory / "benchmarks/flask/validation.csv"
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
