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

def extract_line_number(url: str) -> int:
    """
    Extract line number from GitHub URL.
    
    Args:
        url: GitHub URL with line number
        
    Returns:
        Line number
    """
    match = re.search(r'#L(\d+)', url)
    if match:
        return int(match.group(1))
    return None

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
    # Use the exact same model name as used for embedding creation
    embedding_function = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

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

def extract_end_lines(end_lines: str) -> List[int]:
    """
    Extract end line numbers from a semicolon-separated string.

    Args:
        end_lines: Semicolon-separated string of end line numbers

    Returns:
        List of end line numbers as integers
    """
    if pd.isna(end_lines):
        return []
    return [int(line.strip()) for line in str(end_lines).split(';')]

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
        "retrieved_snippets": [],
        "exact_match_score": []
    }
    
    # Create temporary directory for downloaded files
    temp_dir = tempfile.mkdtemp()
    downloaded_files = []
    
    try:
        for _, row in validation_data.iterrows():
            question = row['Question']
            expected_urls = extract_github_urls(row['URL'])
            end_lines = extract_end_lines(row['end_line'])
            
            # Download and process expected files
            expected_snippets = []
            for i, url in enumerate(expected_urls):
                try:
                    file_path, content = download_github_file(url)
                    downloaded_files.append(file_path)
                    line_number = extract_line_number(url)
                    if line_number:
                        end_line = end_lines[i] if i < len(end_lines) else None
                        snippet = get_code_snippet(content, line_number, end_line)
                        expected_snippets.append(snippet)
                except Exception as e:
                    print(f"Error processing {url}: {str(e)}")
            
            # Get retrieved documents
            retrieved_docs = retriever.invoke(question)
            retrieved_snippets = [doc.page_content for doc in retrieved_docs]

            # calculate scores and store results
            score = exact_match_score(expected_snippets, retrieved_snippets)
            results["question"].append(question)
            results["expected_snippets"].append(expected_snippets)
            results["retrieved_snippets"].append(retrieved_snippets)
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

def exact_match_score(expected_snippets, retrieved_snippets):
    """    Calculate the exact match score between expected and retrieved snippets."""
    matches = sum(1 for exp in expected_snippets
                 if any(exp in ret for ret in retrieved_snippets))
    return matches / len(expected_snippets) if expected_snippets else 0

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
        for j, snippet in enumerate(results["expected_snippets"][i]):
            print(f"  Snippet {j+1}:")
            # Only display first lines to keep output manageable
            snippet_lines = snippet.split("\n")
            snippet_preview = "\n    ".join(snippet_lines[:3])
            if len(snippet_lines) > 3:
                snippet_preview += "\n    ..."
            print(f"    {snippet_preview}")

        print("\nRetrieved Snippets:")
        for j, snippet in enumerate(results["retrieved_snippets"][i][:3]):  # Limit to first 3 results
            print(f"  Snippet {j+1}:")
            snippet_lines = snippet.split("\n")
            snippet_preview = "\n    ".join(snippet_lines[:3])
            if len(snippet_lines) > 3:
                snippet_preview += "\n    ..."
            print(f"    {snippet_preview}")

        if len(results["retrieved_snippets"][i]) > 3:
            print(f"  ... and {len(results['retrieved_snippets'][i]) - 3} more snippets")

def main():
    # Load validation data
    validation_data =pd.read_csv('../../data/validation.csv')
    
    # Load retriever
    retriever = load_retriever()
    
    # Run validation
    results = validate_retriever(retriever, validation_data)
    
    # Print results
    print_validation_results(results)

if __name__ == '__main__':
    main() 
