import os
import glob
import logging
from typing import List, Optional, Dict, Any, Callable, Union

# ChromaDB imports
import chromadb
from chromadb.utils import embedding_functions

# LangChain imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import Language
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings

# import utils
from src.embeddings.utils import create_or_get_collection, add_documents_to_collection
# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_all_python_files(source_dir: str) -> List[str]:
    """
    Recursively find all Python files in the source directory and its subdirectories.

    Args:
        source_dir: Path to the source directory

    Returns:
        List of paths to Python files
    """
    # Use recursive glob pattern to find all .py files
    python_files = glob.glob(os.path.join(source_dir, "**", "*.py"), recursive=True)
    logger.info(f"Found {len(python_files)} Python files in {source_dir}")
    return python_files


def load_and_split_python_files(file_paths: List[str],
                                chunk_size: int = 1000,
                                chunk_overlap: int = 100) -> List[Any]:
    """
    Load and split Python files using the Language-specific splitter.
    Adds byterange and line number information to each chunk's metadata.

    Args:
        file_paths: List of paths to Python files
        chunk_size: Size of text chunks for splitting
        chunk_overlap: Overlap between chunks

    Returns:
        List of document chunks with byterange and line number metadata
    """
    # Initialize Python-specific text splitter
    text_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    all_splits = []

    # Process each Python file
    for i, file_path in enumerate(file_paths):
        try:
            logger.info(f"Processing file {i + 1}/{len(file_paths)}: {file_path}")

            # skip the __init__.py files and cli.py files
            if "__init__.py" in file_path or "cli.py" in file_path:
                logger.info(f"Skipping {file_path}")
                continue

            # Load the Python file
            loader = TextLoader(file_path)
            documents = loader.load()

            # Add metadata to each document
            source_dir = os.path.dirname(os.path.dirname(file_path))  # Adjust based on your structure
            rel_path = os.path.relpath(file_path, source_dir)

            # Get the full content of the file to calculate byte ranges
            with open(file_path, 'rb') as f:
                file_content = f.read().decode('utf-8')
                
            # Split file content into lines for line number calculation
            file_lines = file_content.splitlines()

            for doc in documents:
                doc.metadata["source"] = rel_path
                doc.metadata["file_path"] = file_path

            # Split the documents
            splits = text_splitter.split_documents(documents)

            # Add byterange metadata to each split
            for split in splits:
                content = split.page_content
                # Possible issue: duplicate content in the file
                # Find the start byte of this content in the original file
                start_byte = file_content.find(content)
                if start_byte != -1:
                    end_byte = start_byte + len(content)
                    # Flache Metadaten statt verschachtelter Dictionaries
                    split.metadata["byterange_start"] = start_byte
                    split.metadata["byterange_end"] = end_byte
                    
                    # Calculate line numbers
                    start_line = file_content[:start_byte].count('\n') + 1
                    end_line = start_line + content.count('\n')
                    # Flache Metadaten statt verschachtelter Dictionaries
                    split.metadata["line_start"] = start_line
                    split.metadata["line_end"] = end_line
                else:
                    # If exact match not found (possibly due to whitespace differences)
                    logger.warning(f"Could not find exact byte range for a chunk in {file_path}")
                    split.metadata["byterange_start"] = -1
                    split.metadata["byterange_end"] = -1
                    split.metadata["line_start"] = -1
                    split.metadata["line_end"] = -1

            all_splits.extend(splits)

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")

    logger.info(f"Created {len(all_splits)} chunks from {len(file_paths)} files")
    return all_splits




def embed_python_directory(
        source_dir: str,
        db_path: str,
        collection_name: str = "python_codebase",
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        reset_collection: bool = False,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> None:
    """
    Process all Python files in a directory and create embeddings.

    Args:
        source_dir: Directory containing Python files
        db_path: Path to store ChromaDB
        collection_name: Name for the ChromaDB collection
        chunk_size: Size of code chunks
        chunk_overlap: Overlap between chunks
        reset_collection: Whether to reset an existing collection
        model_name: Name of the embedding model to use
    """
    # Get all Python files
    python_files = get_all_python_files(source_dir)

    if not python_files:
        logger.warning(f"No Python files found in {source_dir}")
        return

    # Load and split Python files
    code_chunks = load_and_split_python_files(
        python_files,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    client = chromadb.PersistentClient(path=db_path)

    # Set up embedding function
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=model_name
    )

    # Create or get collection
    collection = create_or_get_collection(
        client,
        collection_name,
        embedding_function,
        reset=reset_collection
    )

    # Add documents to collection
    add_documents_to_collection(collection, code_chunks, batch_size=1000)

    logger.info(f"Successfully embedded {len(python_files)} Python files into {collection_name}")

if __name__ == "__main__":
    # Example usage
    embed_python_directory(
        source_dir="../../../flask/src",
        db_path="../../chroma_db",
        collection_name="python_codebase",
        chunk_size=1000,
        chunk_overlap=0,
        reset_collection=True,
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

