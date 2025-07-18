import os
import logging
from typing import List, Tuple
import tree_sitter_python as tspython
from tree_sitter import Language, Parser
from langchain_core.documents import Document
from dotenv import load_dotenv

from src.embeddings.file_processor import (
    get_all_python_files,
    get_src_base_path
)
from src.pipelines.embedding_pipeline import EmbeddingPipeline
from src.embeddings.utils import add_documents_to_collection
from src.utils import load_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_function_details(node, source_code: bytes) -> Tuple[str, str, int, int, int, int]:
    """
    Extract name, content, start byte, end byte, start line and end line from a function node.

    Args:
        node: Tree-sitter node representing a function or method
        source_code: Source code as bytes

    Returns:
        Tuple containing (function_name, function_content, start_byte, end_byte, start_line, end_line)
    """
    name_node = node.child_by_field_name('name')
    function_name = name_node.text.decode('utf-8') if name_node else "anonymous_function"

    start_byte = node.start_byte
    end_byte = node.end_byte
    function_content = source_code[start_byte:end_byte].decode('utf-8')

    start_line = node.start_point[0] + 1
    end_line = node.end_point[0] + 1

    return function_name, function_content, start_byte, end_byte, start_line, end_line


def load_and_split_python_functions(
        file_paths: List[str]
) -> List[Document]:
    """
    Load Python files and split them by functions/methods using Tree-sitter.
    Each function or method becomes its own chunk with metadata.

    Args:
        file_paths: List of paths to Python files

    Returns:
        List of document chunks with metadata
    """
    PY_LANGUAGE = Language(tspython.language())
    parser = Parser(PY_LANGUAGE)
    all_chunks = []

    for i, file_path in enumerate(file_paths):
        try:
            logger.info(f"Processing file {i + 1}/{len(file_paths)}: {file_path}")

            # Skip __init__.py and cli.py files
            if "__init__.py" in file_path or "cli.py" in file_path:
                logger.info(f"Skipping {file_path}")
                continue

            with open(file_path, 'rb') as f:
                source_code = f.read()

            tree = parser.parse(source_code)

            source_dir = get_src_base_path(file_path)
            rel_path = os.path.relpath(file_path, source_dir)
            base_metadata = {
                "source": rel_path,
                "file_path": file_path,
            }

            # Query for functions and classes
            query = """
                (function_definition) @function
                (class_definition) @class
            """

            captures = parser.language.query(query).captures(tree.root_node)
            functions = captures["function"]
            classes = captures["class"]
            for node in functions:
                function_name, function_content, start_byte, end_byte, start_line, end_line = extract_function_details(
                    node, source_code
                )

                chunk = Document(
                    page_content=function_content,
                    metadata={
                        **base_metadata,
                        "chunk_type": "function",
                        "function_name": function_name,
                        "byterange_start": start_byte,
                        "byterange_end": end_byte,
                        "line_start": start_line,
                        "line_end": end_line
                    }
                )

                all_chunks.append(chunk)

            for node in classes:
                class_name_node = node.child_by_field_name('name')
                class_name = class_name_node.text.decode('utf-8') if class_name_node else "anonymous_class"

                class_content = source_code[node.start_byte:node.end_byte].decode('utf-8')

                class_chunk = Document(
                    page_content=class_content,
                    metadata={
                        **base_metadata,
                        "chunk_type": "class",
                        "class_name": class_name,
                        "byterange_start": node.start_byte,
                        "byterange_end": node.end_byte,
                        "line_start": node.start_point[0] + 1,
                        "line_end": node.end_point[0] + 1
                    }
                )
                all_chunks.append(class_chunk)

                # Now find and extract all methods within the class
                method_query = """
                        (class_definition body: (block (function_definition) @method))
                    """

                method_captures = parser.language.query(method_query).captures(tree.root_node)
                for method_node in method_captures["method"]:
                    parent_block = method_node.parent
                    if parent_block and parent_block.parent == node:
                        method_name, method_content, m_start_byte, m_end_byte, m_start_line, m_end_line = extract_function_details(
                            method_node, source_code
                        )

                        method_chunk = Document(
                            page_content=method_content,
                            metadata={
                                **base_metadata,
                                "chunk_type": "method",
                                "method_name": method_name,
                                "class_name": class_name,
                                "byterange_start": m_start_byte,
                                "byterange_end": m_end_byte,
                                "line_start": m_start_line,
                                "line_end": m_end_line
                            }
                        )
                        all_chunks.append(method_chunk)

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")

    logger.info(f"Created {len(all_chunks)} function/method chunks from {len(file_paths)} files")
    return all_chunks


def embed_python_functions(
        source_dir: str,
        db_path: str,
        collection_name: str = "python_functions",
        reset_collection: bool = False,
) -> None:
    """
    Process all Python files in a directory and create embeddings at function level
    using the EmbeddingPipeline.

    Args:
        source_dir: Directory containing Python files
        db_path: Path to store ChromaDB
        collection_name: Name for the ChromaDB collection
        include_class_definitions: Whether to include class definitions as chunks
        reset_collection: Whether to reset an existing collection
    """
    # Get all Python files
    python_files = get_all_python_files(source_dir)

    if not python_files:
        logger.warning(f"No Python files found in {source_dir}")
        return

    code_chunks = load_and_split_python_functions(
        python_files
    )

    # Initialize the embedding pipeline
    pipeline = EmbeddingPipeline(
        collection_name=collection_name,
        chroma_persist_directory=db_path,
        reset_collection=reset_collection
    )

    # Add documents to collection using the pipeline
    add_documents_to_collection(
        collection=pipeline.collection,
        documents=code_chunks,
        batch_size=100,
        pipeline=pipeline  # Pass self as the pipeline parameter
    )

    logger.info(
        f"Successfully embedded {len(code_chunks)} function/method chunks from {len(python_files)} Python files in {collection_name}")
