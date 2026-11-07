"""Load pinned questions and validate their source evidence."""

import csv
import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse

from src.index_state import file_hashes

# Dataset identity from the archived retrieval tests.
METRIC_VERSION = "unique-line-coverage-v2"


def extract_file_path_from_url(url):
    parts = urlparse(url).path.split("/", 5)
    if len(parts) != 6 or parts[3] != "blob":
        raise ValueError("Expected a pinned GitHub source URL")
    return parts[5]


def normalize_source_path(source, repository_root):
    root = Path(repository_root).resolve(strict=True)
    path = Path(source.replace("\\", "/"))
    if ".." in path.parts:
        raise ValueError("Source paths must stay inside the repository")
    candidate = (root / path).resolve()
    if not candidate.is_relative_to(root) or not candidate.is_file():
        raise ValueError(
            f"Source file is missing or outside the repository: {source}"
        )
    return candidate.relative_to(root).as_posix()


def validate_labels(row, root, prefix):
    urls = [url.strip() for url in row["URL"].split(";")]
    starts = [int(value) for value in row["start_line"].split(";")]
    ends = [int(value) for value in row["end_line"].split(";")]
    if not urls or len(urls) != len(starts) or len(urls) != len(ends):
        raise ValueError(f"Mismatched label counts for {row['Question']}")
    for url, start, end in zip(urls, starts, ends):
        if not url.startswith(prefix):
            raise ValueError(
                "Every benchmark URL must use the pinned revision"
            )
        path = normalize_source_path(extract_file_path_from_url(url), root)
        count = len((root / path).read_text(encoding="utf-8").splitlines())
        if not 1 <= start <= end <= count:
            raise ValueError(f"Invalid line range in {url}")


def load_benchmark(benchmark_dir, repository_root):
    """Return questions and metadata after validating their sources."""
    directory = Path(benchmark_dir)
    root = Path(repository_root).resolve(strict=True)
    manifest = json.loads((directory / "manifest.json").read_text())
    if (
        manifest["schema_version"] != 1
        or manifest["path_format"] != "repository-relative"
        or manifest["metric_version"] != METRIC_VERSION
    ):
        raise ValueError("Unsupported benchmark manifest format")
    csv_path = directory / "validation.csv"
    if (
        hashlib.sha256(csv_path.read_bytes()).hexdigest()
        != manifest["validation_sha256"]
    ):
        raise ValueError("Benchmark CSV differs from its pinned manifest")
    if file_hashes(root) != manifest["file_hashes"]:
        raise ValueError(
            "Python corpus differs from pinned revision "
            f"{manifest['revision']}"
        )
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != manifest["question_count"]:
        raise ValueError("Benchmark question count differs from its manifest")
    prefix = f"{manifest['repository_url']}/blob/{manifest['revision']}/"
    for row in rows:
        validate_labels(row, root, prefix)
    return {
        "questions": [row["Question"] for row in rows],
        "metadata": {
            key: manifest[key]
            for key in ("dataset_id", "revision", "metric_version")
        },
    }
