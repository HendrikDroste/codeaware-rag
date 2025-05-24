import os
import pandas as pd
import tempfile
import requests
import re
from typing import List, Dict, Any, Tuple
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import VectorStoreRetriever
from urllib.parse import urlparse
from src.utils import load_config
from src.embeddings.embedding_functions import get_embedding_function

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

def load_retriever() -> VectorStoreRetriever:
    """
    Load the retriever for the ChromaDB collection.
    
    Returns:
        VectorStoreRetriever for the ChromaDB collection
    """
    config = load_config("app_config")
    model_name = config['models']['embeddings']['name']
    model_type = config['models']['embeddings']['type']
    model_vendor = config['models']['embeddings']['vendor']
    
    # for_chromadb=False, since we need the LangChain interface here
    embedding_function = get_embedding_function(model_name, model_type, model_vendor, for_chromadb=False)

    # Create ChromaDB client with the same path
    db = Chroma(
        persist_directory="../../chroma_db",
        collection_name="python_codebase",
        embedding_function=embedding_function
    )
    
    # log the number of documents in the collection
    doc_count = db._collection.count()
    print(f"Number of found documents in the collection: {doc_count}")

    # Create the retriever
    retriever = db.as_retriever(search_kwargs={"k": 5})
    return retriever

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
        return '/'.join(path_parts[5:])
    return os.path.basename(parsed_url.path)

def validate_retriever(retriever: VectorStoreRetriever, validation_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate the retriever against the validation data.
    
    Args:
        retriever: The retriever to validate
        validation_data: DataFrame containing validation data
        
    Returns:
        Dictionary containing validation results
    """
    results = {
        "question": [],
        "expected_snippets": [],
        "expected_metadata": [],
        "retrieved_snippets": [],
        "retrieved_metadata": [],
        "exact_match_score": []
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
            expected_snippets = []
            expected_metadata = []
            for i, url in enumerate(expected_urls):
                try:
                    file_path, content = download_github_file(url)
                    downloaded_files.append(file_path)

                    relative_path = extract_file_path_from_url(url)
                    start_line = start_lines[i]
                    if start_line:
                        end_line = end_lines[i] if i < len(end_lines) else start_line
                        snippet = get_code_snippet(content, start_line, end_line)
                        expected_snippets.append(snippet)
                        expected_metadata.append({
                            "filename": relative_path,
                            "line_start": start_line,
                            "line_end": end_line or start_line
                        })
                except Exception as e:
                    print(f"Error processing {url}: {str(e)}")
            
            # Get retrieved documents
            retrieved_docs = retriever.invoke(question)
            retrieved_snippets = [doc.page_content for doc in retrieved_docs]
            retrieved_metadata = [doc.metadata for doc in retrieved_docs]

            # calculate scores and store results
            score = exact_match_score(expected_metadata, retrieved_metadata)
            results["question"].append(question)
            results["expected_snippets"].append(expected_snippets)
            results["expected_metadata"].append(expected_metadata)
            results["retrieved_snippets"].append(retrieved_snippets)
            results["retrieved_metadata"].append(retrieved_metadata)
            results["exact_match_score"].append(score)

    
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

def exact_match_score(expected_metadata, retrieved_metadata):
    """
    Calculate the score based on filename and line number matches.
    
    Args:
        expected_metadata: List of dictionaries with filename, line_start, line_end
        retrieved_metadata: List of dictionaries with metadata from retriever
        
    Returns:
        Float score between 0 and 1
    """
    if not expected_metadata:
        return 0
    
    matches = 0
    for expected in expected_metadata:
        expected_file = expected.get("filename")
        expected_start = expected.get("line_start")
        expected_end = expected.get("line_end")

        # remove the /src/ prefix from the filename
        if expected_file and expected_file.startswith("src/"):
            expected_file = expected_file[4:]
        
        for retrieved in retrieved_metadata:
            retrieved_file = retrieved.get("source") or retrieved.get("filename")
            
            # Check if filenames match (handle potential path differences)
            filename_match = False
            if expected_file and retrieved_file:
                if expected_file == retrieved_file:
                    filename_match = True
                
            if filename_match:
                # If file matches, check for line overlap
                retrieved_start = retrieved.get("line_start")
                retrieved_end = retrieved.get("line_end")
                
                # If no line numbers in metadata, count as a match based on filename only
                if retrieved_start is None or retrieved_end is None:
                    matches += 1
                    break

                # Check for line range overlap
                if (retrieved_start <= expected_end and retrieved_end >= expected_start):
                    matches += 1
                    break

    return matches / len(expected_metadata)

def print_validation_results(results: Dict[str, Any]):
    """
    Print validation results in a readable format.
    
    Args:
        results: Dictionary containing validation results
    """
    # Calculate overall accuracy
    avg_score = sum(results["exact_match_score"]) / len(results["exact_match_score"]) if results["exact_match_score"] else 0

    print("\n=== Validation Results ===")
    print(f"Average Exact Match Score: {avg_score:.2f} ({sum(results['exact_match_score'])}/{len(results['exact_match_score'])} matches)")

    # Detailed output for each question
    for i in range(len(results["question"])):
        print(f"\n--- Question {i+1} ---")
        print(f"Question: {results['question'][i]}")
        print(f"Exact Match Score: {results['exact_match_score'][i]:.2f}")

        print("\nExpected Snippets:")
        for j, (snippet, metadata) in enumerate(zip(results["expected_snippets"][i], results["expected_metadata"][i])):
            print(f"  Snippet {j+1} ({metadata['filename']}, lines {metadata['line_start']}-{metadata['line_end']}):")
            # Only display first lines to keep output manageable
            snippet_lines = snippet.split("\n")
            snippet_preview = "\n    ".join(snippet_lines[:3])
            if len(snippet_lines) > 3:
                snippet_preview += "\n    ..."
            print(f"    {snippet_preview}")

        print("\nRetrieved Snippets:")
        for j, (snippet, metadata) in enumerate(zip(results["retrieved_snippets"][i][:3], results["retrieved_metadata"][i][:3])):  # Limit to first 3 results
            filename = metadata.get("source", metadata.get("filename", "unknown"))
            line_start = metadata.get("line_start", "?")
            line_end = metadata.get("line_end", "?")
            print(f"  Snippet {j+1} ({filename}, lines {line_start}-{line_end}):")
            snippet_lines = snippet.split("\n")
            snippet_preview = "\n    ".join(snippet_lines[:3])
            if len(snippet_lines) > 3:
                snippet_preview += "\n    ..."
            print(f"    {snippet_preview}")

        if len(results["retrieved_snippets"][i]) > 3:
            print(f"  ... and {len(results['retrieved_snippets'][i]) - 3} more snippets")

    # print total score
    print(f"\nTotal Exact Match Score: {sum(results['exact_match_score'])}/{len(results['exact_match_score'])} ({avg_score:.2f})")

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
        question_score = results["exact_match_score"][i]
        expected_metadata = results["expected_metadata"][i]
        
        # Format expected metadata for reference
        expected_files = [f"{meta.get('filename')}:{meta.get('line_start')}-{meta.get('line_end')}" 
                         for meta in expected_metadata]
        expected_files_str = "; ".join(expected_files)
        
        # Add rows for each retrieved snippet
        for j, (snippet, metadata) in enumerate(zip(results["retrieved_snippets"][i], results["retrieved_metadata"][i])):
            filename = metadata.get("source", metadata.get("filename", "unknown"))
            line_start = metadata.get("line_start", "")
            line_end = metadata.get("line_end", "")
            
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
                "question_score": question_score,
                "expected_files": expected_files_str,
                "snippet": snippet
            })
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Ergebnisse wurden in {output_path} gespeichert")

def main():
    # Load validation data
    validation_data = pd.read_csv('../../data/validation.csv')
    
    # Load retriever
    retriever = load_retriever()
    
    # Run validation
    results = validate_retriever(retriever, validation_data)
    
    # Print results
    print_validation_results(results)
    
    # Save results to CSV
    #save_results_to_csv(results)

if __name__ == '__main__':
    main()
