import os
import logging
from typing import List, Tuple, Optional
import tree_sitter_python as tspython
from tree_sitter import Language, Parser
from langchain_core.documents import Document
from dotenv import load_dotenv

from src.embeddings.file_processor import (
    get_all_python_files,
    get_src_base_path
)
from src.pipelines.embedding_pipeline import EmbeddingPipeline
from src.utils import load_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_docstring(node, source_code: bytes) -> Optional[str]:
    """
    Extract docstring from a function or class node.

    Args:
        node: Tree-sitter node representing a function or class
        source_code: Source code as bytes

    Returns:
        Docstring if found, None otherwise
    """
    # Look for the body of the function/class
    body = node.child_by_field_name('body')
    if not body:
        return None

    # Check if first statement is an expression containing a string
    if body.children and len(body.children) > 0:
        first_child = body.children[0]

        # Check if it's an expression statement with a string literal
        if first_child.type == "expression_statement":
            expr = first_child.children[0] if first_child.children else None

            if expr and expr.type == "string":
                docstring = source_code[expr.start_byte:expr.end_byte].decode('utf-8')
                return docstring

    return None


def extract_documentation_details(node, source_code: bytes) -> Tuple[Optional[str], str, int, int, int, int]:
    """
    Extract documentation, content, start byte, end byte, start line and end line from a node.

    Args:
        node: Tree-sitter node representing a function or method
        source_code: Source code as bytes

    Returns:
        Tuple containing (documentation, node_content, start_byte, end_byte, start_line, end_line)
    """
    name_node = node.child_by_field_name('name')
    node_name = name_node.text.decode('utf-8') if name_node else "anonymous_entity"

    start_byte = node.start_byte
    end_byte = node.end_byte
    node_content = source_code[start_byte:end_byte].decode('utf-8')

    start_line = node.start_point[0] + 1
    end_line = node.end_point[0] + 1

    # Extract docstring
    documentation = extract_docstring(node, source_code)

    return (documentation, node_content, start_byte, end_byte, start_line, end_line)


def load_and_extract_python_documentation(file_paths: List[str]) -> List[Document]:
    """
    Load Python files and extract documentation using Tree-sitter.
    Creates document chunks from docstrings of functions, methods, and classes.

    Args:
        file_paths: List of paths to Python files

    Returns:
        List of document chunks containing documentation with metadata
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
            functions = captures["function"] if "function" in captures else []
            classes = captures["class"] if "class" in captures else []

            # Process functions
            for node in functions:
                doc, content, start_byte, end_byte, start_line, end_line = extract_documentation_details(node, source_code)

                if doc:  # Only create a chunk if documentation is available
                    name_node = node.child_by_field_name('name')
                    function_name = name_node.text.decode('utf-8') if name_node else "anonymous_function"

                    chunk = Document(
                        page_content=doc,
                        metadata={
                            **base_metadata,
                            "chunk_type": "function_doc",
                            "function_name": function_name,
                            "byterange_start": start_byte,
                            "byterange_end": end_byte,
                            "line_start": start_line,
                            "line_end": end_line
                        }
                    )
                    all_chunks.append(chunk)

            # Process classes and their methods
            for node in classes:
                class_name_node = node.child_by_field_name('name')
                class_name = class_name_node.text.decode('utf-8') if class_name_node else "anonymous_class"

                # Extract class docstring
                class_doc, class_content, c_start_byte, c_end_byte, c_start_line, c_end_line = extract_documentation_details(node, source_code)

                if class_doc:
                    class_chunk = Document(
                        page_content=class_doc,
                        metadata={
                            **base_metadata,
                            "chunk_type": "class_doc",
                            "class_name": class_name,
                            "byterange_start": c_start_byte,
                            "byterange_end": c_end_byte,
                            "line_start": c_start_line,
                            "line_end": c_end_line
                        }
                    )
                    all_chunks.append(class_chunk)

                # Find and extract all methods and their docstrings within the class
                method_query = """
                    (class_definition body: (block (function_definition) @method))
                """

                method_captures = parser.language.query(method_query).captures(tree.root_node)
                for method_node in method_captures["method"]:
                    parent_block = method_node.parent
                    if parent_block and parent_block.parent == node:
                        method_doc, method_content, m_start_byte, m_end_byte, m_start_line, m_end_line = extract_documentation_details(method_node, source_code)

                        if method_doc:
                            method_name_node = method_node.child_by_field_name('name')
                            method_name = method_name_node.text.decode('utf-8') if method_name_node else "anonymous_method"

                            method_chunk = Document(
                                page_content=method_doc,
                                metadata={
                                    **base_metadata,
                                    "chunk_type": "method_doc",
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

    logger.info(f"Created {len(all_chunks)} documentation chunks from {len(file_paths)} files")
    return all_chunks


def embed_python_documentation(
        source_dir: str,
        db_path: str,
        collection_name: str = "python_documentation",
        reset_collection: bool = False,
) -> None:
    """
    Process all Python files in a directory and create embeddings of their documentation
    using the EmbeddingPipeline.

    Args:
        source_dir: Directory containing Python files
        db_path: Path to store ChromaDB
        collection_name: Name for the ChromaDB collection
        reset_collection: Whether to reset an existing collection
    """
    # Get all Python files
    python_files = get_all_python_files(source_dir)

    if not python_files:
        logger.warning(f"No Python files found in {source_dir}")
        return

    # Extract documentation chunks
    doc_chunks = load_and_extract_python_documentation(python_files)

    if not doc_chunks:
        logger.warning("No documentation found in Python files")
        return

    # Initialize the embedding pipeline
    pipeline = EmbeddingPipeline(
        collection_name=collection_name,
        chroma_persist_directory=db_path,
        reset_collection=reset_collection
    )

    # Add documents to collection using the pipeline
    pipeline.add_documents(doc_chunks, batch_size=1)

    logger.info(
        f"Successfully embedded {len(doc_chunks)} documentation chunks from {len(python_files)} Python files in {collection_name}")


if __name__ == "__main__":
    load_dotenv()
    config = load_config("app_config")

    embed_python_documentation(
        source_dir="../../../flask/src",
        db_path="../../chroma_db",
        collection_name="python_documentation",
        reset_collection=True
    )

