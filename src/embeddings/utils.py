import os
import logging
from typing import List, Optional, Dict, Any, Callable, Union, TYPE_CHECKING

# ChromaDB imports
import chromadb
from chromadb.utils import embedding_functions

if TYPE_CHECKING:
    from src.pipelines.base_pipeline import BaseRAGPipeline

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_or_get_collection(client: chromadb.Client,
                             collection_name: str,
                             embedding_function: Any,
                             reset: bool = False) -> Any:
    """
    Create a new ChromaDB collection or get existing one.

    Args:
        client: ChromaDB client
        collection_name: Name of the collection
        embedding_function: Function to use for embeddings
        reset: Whether to reset an existing collection

    Returns:
        ChromaDB collection
    """
    # Check if collection exists and delete it if specified
    if reset:
        try:
            client.delete_collection(collection_name)
            logger.info(f"Deleted existing collection '{collection_name}'")
        except Exception as e:
            logger.info(f"No existing collection to delete: {e}")

    try:
        # Try to get existing collection
        collection = client.get_collection(
            name=collection_name,
            embedding_function=embedding_function
        )
        logger.info(f"Using existing collection '{collection_name}'")
    except:
        # Create new collection if it doesn't exist
        collection = client.create_collection(
            name=collection_name,
            embedding_function=embedding_function,
            metadata={"description": f"Collection for {collection_name} embeddings"}
        )
        logger.info(f"Created new collection '{collection_name}'")

    return collection


def add_documents_to_collection(collection: Any,
                                documents: List[Any],
                                batch_size: int = 100,
                                id_prefix: str = "doc",
                                extra_metadata: Optional[Dict[str, Any]] = None,
                                metadata_fn: Optional[Callable[[Dict[str, Any], int], Dict[str, Any]]] = None,
                                pipeline: Optional["BaseRAGPipeline"] = None) -> None:
    """
    Add documents to a ChromaDB collection in batches with enhanced metadata.

    Args:
        collection: ChromaDB collection
        documents: List of LangChain documents
        batch_size: Number of documents to process in each batch
        id_prefix: Prefix for document IDs
        extra_metadata: Additional metadata to add to all documents
        metadata_fn: Optional function to transform metadata (takes metadata dict and index as input)
        pipeline: Optional BaseRAGPipeline to use for generating embeddings
    """
    logger.info(f"Adding {len(documents)} documents to ChromaDB collection")

    # Convert LangChain documents to format expected by ChromaDB
    ids = []
    texts = []
    metadatas = []

    for i, doc in enumerate(documents):
        # Create document ID
        doc_id = f"{id_prefix}_{i}"
        ids.append(doc_id)

        # Add document text
        texts.append(doc.page_content)

        # Process metadata
        metadata = doc.metadata.copy()

        # Add document index as metadata
        metadata["doc_index"] = i
        metadata["doc_id"] = doc_id

        # Convert byterange dict to string if it exists
        if "byterange" in metadata and isinstance(metadata["byterange"], dict):
            start = metadata["byterange"].get("start")
            end = metadata["byterange"].get("end")
            metadata["byterange_start"] = start
            metadata["byterange_end"] = end
            del metadata["byterange"]  # Remove the dictionary

        # Add any extra metadata to all documents
        if extra_metadata:
            metadata.update(extra_metadata)

        # Apply custom metadata transformation if provided
        if metadata_fn:
            metadata = metadata_fn(metadata, i)

        metadatas.append(metadata)

    # Use batch processing to add documents
    for i in range(0, len(texts), batch_size):
        end_idx = min(i + batch_size, len(texts))

        batch_ids = ids[i:end_idx]
        batch_texts = texts[i:end_idx]
        batch_metadatas = metadatas[i:end_idx]

        # Add documents with embeddings from pipeline if available
        if pipeline:
            # Generate embeddings using the pipeline's prepare_batch function
            embeddings = pipeline.prepare_batch(batch_texts)

            collection.add(
                ids=batch_ids,
                documents=batch_texts,
                metadatas=batch_metadatas,
                embeddings=embeddings
            )
            logger.info(f"Added batch {i // batch_size + 1} with pipeline embeddings")
        else:
            # Use ChromaDB's default embedding function
            collection.add(
                ids=batch_ids,
                documents=batch_texts,
                metadatas=batch_metadatas
            )
            logger.info(f"Added batch {i // batch_size + 1} with ChromaDB embeddings")

    # Verify data was added
    count = collection.count()
    logger.info(f"Collection now contains {count} documents")