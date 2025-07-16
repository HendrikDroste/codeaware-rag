import os
import pandas as pd
import tempfile
import requests
import datetime
from typing import List, Dict, Any, Tuple
from urllib.parse import urlparse
from src.pipelines.base_pipeline import BaseRAGPipeline
from src.retrievers.metrics import (
    mean_reciprocal_rank,
    line_coverage_ratio,
    file_match_ratio,
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
    raw_url = url.replace('github.com', 'raw.githubusercontent.com').replace('/blob/', '/')

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

def get_code_snippet(content: str, line_number: int, end_line: int = None, context_lines: int = 0) -> str:
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

    # If end_line is specified, use it; otherwise, default to start_idx + 1
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

    if relative_path.startswith("src/"):
        relative_path = relative_path[4:]
    return relative_path

def validate_retriever(pipeline: BaseRAGPipeline, validation_data: pd.DataFrame) -> dict[str, list[Any]] | None:
    """
    Validate the retriever against the validation data.

    Args:
        pipeline: The embedding pipeline to validate
        validation_data: DataFrame containing validation data

    Returns:
        Dictionary containing validation results
    """
    results = {
        "question": [],
        "expected_metadata": [],
        "retrieved_metadata": [],
        "file_match_ratio": [],
        "mrr": [],
        "line_coverage": []
    }

    # Create temporary directory for downloaded files
    temp_dir = tempfile.mkdtemp()
    downloaded_files = []

    try:
        for _, row in validation_data.iterrows():
            question = row['Question']
            expected_urls = extract_github_urls(row['URL'])
            end_lines = extract_lines(row['end_line'])
            start_lines = extract_lines(row['start_line'])

            # Download and process expected files
            expected_metadata = []
            for i, url in enumerate(expected_urls):
                try:
                    file_path, content = download_github_file(url)
                    downloaded_files.append(file_path)

                    relative_path = extract_file_path_from_url(url)
                    start_line = start_lines[i]
                    if start_line:
                        end_line = end_lines[i] if i < len(end_lines) else start_line
                        expected_metadata.append({
                            "filename": relative_path,
                            "line_start": start_line,
                            "line_end": end_line or start_line
                        })
                except Exception as e:
                    print(f"Error processing {url}: {str(e)}")

            # Get retrieved documents using the pipeline
            pipeline_results = pipeline.invoke(question)
            
            # Convert pipeline results to a format compatible with our metrics
            retrieved_metadata = []
            for i in range(len(pipeline_results["ids"][0])):
                metadata = pipeline_results["metadatas"][0][i]
                # Ensure metadata has the required keys
                metadata_entry = {
                    "source": metadata.get("source", ""),
                    "filename": metadata.get("filename", ""),
                    "line_start": metadata.get("line_start", 0),
                    "line_end": metadata.get("line_end", 0),
                    "content": pipeline_results["documents"][0][i]
                }
                retrieved_metadata.append(metadata_entry)

            # calculate scores using imported metrics
            mrr = mean_reciprocal_rank(expected_metadata, retrieved_metadata)
            line_cov = line_coverage_ratio(expected_metadata, retrieved_metadata)
            ems = file_match_ratio(expected_metadata, retrieved_metadata)

            # store results
            results["question"].append(question)
            results["expected_metadata"].append(expected_metadata)
            results["retrieved_metadata"].append(retrieved_metadata)
            results["file_match_ratio"].append(ems)
            results["mrr"].append(mrr)
            results["line_coverage"].append(line_cov)

    finally:
        # Clean up temporary files
        for file_path in downloaded_files:
            try:
                os.remove(file_path)
            except:
                pass
        try:
            os.rmdir(temp_dir)
        except:
            pass

    return results


def print_validation_results(results: Dict[str, Any]):
    """
    Print validation results in a readable format.

    Args:
        results: Dictionary containing validation results
    """
    # Calculate overall metrics
    avg_exact_match = sum(results["file_match_ratio"]) / len(results["file_match_ratio"]) if results["file_match_ratio"] else 0
    avg_mrr = sum(results["mrr"]) / len(results["mrr"]) if results["mrr"] else 0
    avg_line_coverage = sum(results["line_coverage"]) / len(results["line_coverage"]) if results["line_coverage"] else 0

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
            print(f"  File {j+1}: {metadata['filename']}, lines {metadata['line_start']}-{metadata['line_end']}")

        print("\nRetrieved Files:")
        for j, metadata in enumerate(results["retrieved_metadata"][i][:3]):  # Limit to first 3 results
            filename = metadata.get("source", metadata.get("filename", "unknown"))
            line_start = metadata.get("line_start", "?")
            line_end = metadata.get("line_end", "?")
            print(f"  File {j+1}: {filename}, lines {line_start}-{line_end}")

        if len(results["retrieved_metadata"][i]) > 3:
            print(f"  ... and {len(results['retrieved_metadata'][i]) - 3} more files")

def save_results_to_csv(results: Dict[str, Any], output_path: str = "../../data/retrieval_results.csv"):
    """
    Save validation results to a CSV file with one row per retrieved snippet.

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
        expected_files = [f"{meta.get('filename')}:{meta.get('line_start')}-{meta.get('line_end')}"
                         for meta in expected_metadata]
        expected_files_str = "; ".join(expected_files)

        # Add rows for each retrieved snippet
        for j, metadata in enumerate(results["retrieved_metadata"][i]):
            filename = metadata.get("source", metadata.get("filename", "unknown"))
            line_start = metadata.get("line_start", "")
            line_end = metadata.get("line_end", "")
            snippet = metadata.get("content", "")

            # Check if this specific snippet matches any expected snippet
            is_match = False
            for exp_meta in expected_metadata:
                exp_file = exp_meta.get("filename")
                # Remove /src/ prefix if present
                if exp_file and exp_file.startswith("src/"):
                    exp_file = exp_file[4:]

                if exp_file == filename:
                    exp_start = exp_meta.get("line_start")
                    exp_end = exp_meta.get("line_end")

                    # Check for line overlap
                    if (line_start is None or line_end is None or
                        (line_start <= exp_end and line_end >= exp_start)):
                        is_match = True
                        break

            rows.append({
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
                "snippet": snippet
            })

    # Create DataFrame and save to CSV
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

def save_model_results(results: Dict[str, Any], model_name: str, output_path: str = "../../data/model_results.csv"):
    """
    Save model validation metrics to the results CSV file.

    Args:
        results: Dictionary containing validation results
        model_name: Name of the model/retriever being evaluated
        output_path: Path to the model_results.csv file
    """
    # Calculate overall metrics
    avg_exact_match = sum(results["file_match_ratio"]) / len(results["file_match_ratio"]) if results["file_match_ratio"] else 0
    avg_mrr = sum(results["mrr"]) / len(results["mrr"]) if results["mrr"] else 0
    avg_line_coverage = sum(results["line_coverage"]) / len(results["line_coverage"]) if results["line_coverage"] else 0

    # Get current date and time in a human-readable format
    current_datetime = datetime.datetime.now().strftime("%d.%m.%Y %H:%M")

    # Create DataFrame with the new row
    new_row = pd.DataFrame({
        "Name": [model_name],
        "file_match_ratio": [avg_exact_match],
        "MRR": [avg_mrr],
        "Line Coverage": [avg_line_coverage],
        "Date": [current_datetime]
    })


    df = pd.read_csv(output_path)
    df = pd.concat([df, new_row], ignore_index=True)

    # Save updated DataFrame
    df.to_csv(output_path, index=False)
    print(f"Model results saved to {output_path}")
