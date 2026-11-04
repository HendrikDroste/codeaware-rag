import datetime
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse

import pandas as pd
import requests

from src.pipelines.base_pipeline import BaseRAGPipeline
from src.retrievers.metrics import (
    METRIC_VERSION,
    file_match_ratio,
    line_coverage_ratio,
    mean_reciprocal_rank,
)


def download_github_file(url: str) -> Tuple[str, str]:
    """
    Download a file from GitHub and return its content and file path.

    Args:
        url: GitHub URL to download

    Returns:
        Tuple of (file_path, content)
    """
    # Convert GitHub URL to raw content URL
    raw_url = url.replace('github.com', 'raw.githubusercontent.com').replace(
        '/blob/', '/'
    )

    # Create temporary file
    temp_dir = tempfile.mkdtemp()
    parsed_url = urlparse(url)
    file_name = os.path.basename(parsed_url.path)
    file_path = os.path.join(temp_dir, file_name)

    # Download file
    response = requests.get(raw_url)
    if response.status_code == 200:
        content = response.text
        with open(file_path, 'w') as f:
            f.write(content)
        return file_path, content
    else:
        raise Exception(f"Failed to download file from {url}")


def get_code_snippet(
    content: str,
    line_number: int,
    end_line: int = None,
    context_lines: int = 0,
) -> str:
    """
    Extract code snippet around the specified line number.

    Args:
        content: File content
        line_number: Target line number
        end_line: End line number (optional)
        context_lines: Number of context lines before and after

    Returns:
        Code snippet
    """
    lines = content.split('\n')

    # Convert line numbers to 0-based indices
    start_idx = max(0, line_number - 1)

    # If end_line is specified, use it; otherwise, default to start_idx
    # + 1
    end_idx = end_line if end_line is not None else start_idx + 1

    # Add context lines
    final_start = max(0, start_idx - context_lines)
    final_end = min(len(lines), end_idx + context_lines)

    return '\n'.join(lines[final_start:final_end])


def extract_github_urls(urls: str) -> List[str]:
    """
    Extract GitHub URLs from a semicolon-separated string.

    Args:
        urls: Semicolon-separated string of URLs

    Returns:
        List of GitHub URLs
    """
    if pd.isna(urls):
        return []
    return [url.strip() for url in urls.split(';')]


def extract_lines(lines: str) -> List[int]:
    """
    Extract line numbers from a semicolon-separated string.

    Args:
        lines: Semicolon-separated string of line numbers

    Returns:
        List of end line numbers as integers
    """
    if pd.isna(lines):
        return []
    return [int(line.strip()) for line in str(lines).split(';')]


def extract_file_path_from_url(url: str) -> str:
    """
    Extract repository-relative file path from GitHub URL.

    Args:
        url: GitHub URL to a file

    Returns:
        Repository-relative file path
    """
    parsed_url = urlparse(url)
    path_parts = parsed_url.path.split('/')

    # Standard GitHub URL format: /:owner/:repo/blob/:branch/:path
    if len(path_parts) > 4 and path_parts[3] == 'blob':
        relative_path = '/'.join(path_parts[5:])
    else:
        relative_path = os.path.basename(parsed_url.path)

    return relative_path


def normalize_source_path(source: str, repository_root) -> str:
    """Normalize paths, including the legacy src-relative convention."""
    root = Path(repository_root).resolve(strict=True)
    path = Path(source.replace("\\", "/"))
    if ".." in path.parts:
        raise ValueError("Source paths must stay inside the repository")
    candidate = (root / path).resolve()
    if not path.is_absolute() and not candidate.is_file():
        candidate = (root / "src" / path).resolve()
    if not candidate.is_relative_to(root) or not candidate.is_file():
        raise ValueError(
            f"Source file is missing or outside the repository: {source}"
        )
    return candidate.relative_to(root).as_posix()


def read_expected_metadata(row, repository_root=None):
    """Read aligned, inclusive labels; reject malformed annotations."""
    urls = extract_github_urls(row["URL"])
    starts, ends = extract_lines(row["start_line"]), extract_lines(
        row["end_line"]
    )
    if not urls or len(urls) != len(starts) or len(urls) != len(ends):
        raise ValueError(f"Mismatched label counts for {row['Question']}")
    labels = []
    for url, start, end in zip(urls, starts, ends):
        path = extract_file_path_from_url(url)
        if start < 1 or end < start:
            raise ValueError(f"Invalid line range in {url}")
        if repository_root is not None:
            path = normalize_source_path(path, repository_root)
            count = len(
                (Path(repository_root) / path)
                .read_text(encoding="utf-8")
                .splitlines()
            )
            if end > count:
                raise ValueError(f"Label extends past the end of {path}")
        elif path.startswith("src/"):
            # Legacy experiments store paths relative to src.
            path = path[4:]
        label = {"filename": path, "line_start": start, "line_end": end}
        if label not in labels:
            labels.append(label)
    return labels


def load_benchmark(benchmark_dir, repository_root) -> pd.DataFrame:
    """Load pinned labels when CSV and Python corpus hashes match."""
    from src.index_state import file_hashes

    directory, root = Path(benchmark_dir), Path(repository_root).resolve(
        strict=True
    )
    manifest = json.loads((directory / "manifest.json").read_text())
    if (
        manifest["schema_version"] != 1
        or manifest["path_format"] != "repository-relative"
    ):
        raise ValueError("Unsupported benchmark manifest format")
    if manifest["metric_version"] != METRIC_VERSION:
        raise ValueError(
            "Benchmark metric version does not match this evaluator"
        )
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
    data = pd.read_csv(csv_path, keep_default_na=False)
    if len(data) != manifest["question_count"]:
        raise ValueError("Benchmark question count differs from its manifest")
    prefix = f"{manifest['repository_url']}/blob/{manifest['revision']}/"
    for _, row in data.iterrows():
        if any(
            not url.startswith(prefix)
            for url in extract_github_urls(row["URL"])
        ):
            raise ValueError(
                "Every benchmark URL must use the pinned revision"
            )
        read_expected_metadata(row, root)
    data.attrs["repository_root"] = str(root)
    data.attrs["file_hashes"] = manifest["file_hashes"]
    data.attrs["benchmark"] = {
        key: manifest[key]
        for key in ("dataset_id", "revision", "metric_version")
    }
    return data


def validate_retriever(
    pipeline: BaseRAGPipeline,
    validation_data: pd.DataFrame,
    *,
    repository_root=None,
) -> dict[str, Any]:
    """
    Validate the retriever against the validation data.

    Args:
        pipeline: The embedding pipeline to validate
        validation_data: DataFrame containing validation data
        repository_root: Local checkout for canonical paths and range
            checks; inferred automatically when using load_benchmark

    Returns:
        Dictionary containing validation results
    """
    results = {
        "question": [],
        "expected_metadata": [],
        "retrieved_metadata": [],
        "file_match_ratio": [],
        "mrr": [],
        "line_coverage": [],
    }

    from src.index_state import file_hashes

    repository_root = repository_root or validation_data.attrs.get(
        "repository_root"
    )
    if repository_root is not None:
        repository_root = Path(repository_root).resolve(strict=True)
    pinned_hashes = validation_data.attrs.get("file_hashes")

    def check_corpus():
        if (
            pinned_hashes is not None
            and file_hashes(repository_root) != pinned_hashes
        ):
            raise ValueError(
                "Python corpus changed after loading the benchmark"
            )

    check_corpus()
    results["benchmark"] = {"metric_version": METRIC_VERSION}
    results["benchmark"].update(validation_data.attrs.get("benchmark", {}))
    # Validate all labels before making expensive model calls.
    labels = [
        read_expected_metadata(row, repository_root)
        for _, row in validation_data.iterrows()
    ]
    for (_, row), expected in zip(validation_data.iterrows(), labels):
        question = row["Question"]
        pipeline_results = pipeline.invoke(question)
        retrieved = []
        for document, metadata in zip(
            pipeline_results["documents"][0], pipeline_results["metadatas"][0]
        ):
            source_path = metadata.get("source") or metadata.get(
                "filename", ""
            )
            if repository_root is not None:
                source_path = normalize_source_path(
                    source_path, repository_root
                )
            retrieved.append(
                {
                    "source": source_path,
                    "filename": source_path,
                    "line_start": metadata.get("line_start"),
                    "line_end": metadata.get("line_end"),
                    "content": document,
                }
            )
        results["question"].append(question)
        results["expected_metadata"].append(expected)
        results["retrieved_metadata"].append(retrieved)
        results["mrr"].append(mean_reciprocal_rank(expected, retrieved))
        results["line_coverage"].append(
            line_coverage_ratio(expected, retrieved)
        )
        results["file_match_ratio"].append(
            file_match_ratio(expected, retrieved)
        )
    check_corpus()
    return results


def print_validation_results(results: Dict[str, Any]):
    """
    Print validation results in a readable format.

    Args:
        results: Dictionary containing validation results
    """
    # Calculate overall metrics
    avg_exact_match = (
        sum(results["file_match_ratio"]) / len(results["file_match_ratio"])
        if results["file_match_ratio"]
        else 0
    )
    avg_mrr = (
        sum(results["mrr"]) / len(results["mrr"]) if results["mrr"] else 0
    )
    avg_line_coverage = (
        sum(results["line_coverage"]) / len(results["line_coverage"])
        if results["line_coverage"]
        else 0
    )

    print("\n=== Validation Results ===")
    print(f"Average Exact Match Score: {avg_exact_match:.2f}")
    print(f"Average Mean Reciprocal Rank: {avg_mrr:.2f}")
    print(f"Average Line Coverage: {avg_line_coverage:.2f}")

    # Detailed output for each question
    for i in range(len(results["question"])):
        print(f"\n--- Question {i+1} ---")
        print(f"Question: {results['question'][i]}")
        print(f"MRR: {results['mrr'][i]:.2f}")
        print(f"Line Coverage: {results['line_coverage'][i]:.2f}")

        print("\nExpected Files:")
        for j, metadata in enumerate(results["expected_metadata"][i]):
            print(
                f"  File {j + 1}: {metadata['filename']}, "
                f"lines {metadata['line_start']}-{metadata['line_end']}"
            )

        print("\nRetrieved Files:")
        for j, metadata in enumerate(
            results["retrieved_metadata"][i][:3]
        ):  # Limit to first 3 results
            filename = metadata.get(
                "source", metadata.get("filename", "unknown")
            )
            line_start = metadata.get("line_start", "?")
            line_end = metadata.get("line_end", "?")
            print(f"  File {j+1}: {filename}, lines {line_start}-{line_end}")

        if len(results["retrieved_metadata"][i]) > 3:
            print(
                f"  ... and {len(results['retrieved_metadata'][i]) - 3} "
                "more files"
            )


def save_results_to_csv(
    results: Dict[str, Any],
    output_path: str = "../../data/retrieval_results.csv",
):
    """
    Save validation results to a CSV file with one row per retrieved
    snippet.

    Args:
        results: Dictionary containing validation results
        output_path: Path where the CSV file should be saved
    """
    rows = []

    for i in range(len(results["question"])):
        question = results["question"][i]
        question_score = results["file_match_ratio"][i]
        mrr = results["mrr"][i]
        line_cov = results["line_coverage"][i]
        expected_metadata = results["expected_metadata"][i]

        # Format expected metadata for reference
        expected_files = [
            f"{meta.get('filename')}:"
            f"{meta.get('line_start')}-{meta.get('line_end')}"
            for meta in expected_metadata
        ]
        expected_files_str = "; ".join(expected_files)

        # Add rows for each retrieved snippet
        for j, metadata in enumerate(results["retrieved_metadata"][i]):
            filename = metadata.get(
                "source", metadata.get("filename", "unknown")
            )
            line_start = metadata.get("line_start", "")
            line_end = metadata.get("line_end", "")
            snippet = metadata.get("content", "")

            # Check if this specific snippet matches any expected
            # snippet
            is_match = False
            for exp_meta in expected_metadata:
                exp_file = exp_meta.get("filename")
                if exp_file == filename:
                    exp_start = exp_meta.get("line_start")
                    exp_end = exp_meta.get("line_end")

                    # Check for line overlap
                    if (
                        line_start is None
                        or line_end is None
                        or (line_start <= exp_end and line_end >= exp_start)
                    ):
                        is_match = True
                        break

            rows.append(
                {
                    "question_id": i + 1,
                    "question": question,
                    "snippet_rank": j + 1,
                    "filename": filename,
                    "line_start": line_start,
                    "line_end": line_end,
                    "is_match": is_match,
                    "file_match_ratio": question_score,
                    "mrr": mrr,
                    "line_coverage": line_cov,
                    "expected_files": expected_files_str,
                    "snippet": snippet,
                    **results.get("benchmark", {}),
                }
            )

    # Create DataFrame and save to CSV
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")


def save_model_results(
    results: Dict[str, Any],
    model_name: str,
    output_path: str = "../../data/model_results.csv",
):
    """
    Save model validation metrics to the results CSV file.

    Args:
        results: Dictionary containing validation results
        model_name: Name of the model/retriever being evaluated
        output_path: Path to the model_results.csv file
    """
    # Calculate overall metrics
    avg_exact_match = (
        sum(results["file_match_ratio"]) / len(results["file_match_ratio"])
        if results["file_match_ratio"]
        else 0
    )
    avg_mrr = (
        sum(results["mrr"]) / len(results["mrr"]) if results["mrr"] else 0
    )
    avg_line_coverage = (
        sum(results["line_coverage"]) / len(results["line_coverage"])
        if results["line_coverage"]
        else 0
    )

    # Get current date and time in a human-readable format
    current_datetime = datetime.datetime.now().strftime("%d.%m.%Y %H:%M")

    # Create DataFrame with the new row
    new_row = pd.DataFrame(
        {
            "Name": [model_name],
            "file_match_ratio": [avg_exact_match],
            "MRR": [avg_mrr],
            "Line Coverage": [avg_line_coverage],
            "Date": [current_datetime],
            **{
                key: [value]
                for key, value in results.get("benchmark", {}).items()
            },
        }
    )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(path) if path.exists() else pd.DataFrame()
    df = pd.concat([df, new_row], ignore_index=True)

    # Save updated DataFrame
    df.to_csv(output_path, index=False)
    print(f"Model results saved to {output_path}")
