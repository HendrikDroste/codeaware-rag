import argparse
from typing import List, Optional
import logging
from datasets import load_dataset
import chromadb
from chromadb.utils import embedding_functions

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_codesearchnet_dataset(split: str = "train", max_samples: Optional[int] = None):
    """
    Load the codesearchnet dataset from Hugging Face.

    Args:
        split: Dataset split to load ('train', 'test', or 'validation')
        max_samples: Maximum number of samples to load (None for all)

    Returns:
        The loaded dataset
    """
    logger.info(f"Loading codesearchnet dataset (split={split})")
    dataset = load_dataset("sentence-transformers/codesearchnet", split=split)

    if max_samples is not None and max_samples < len(dataset):
        dataset = dataset.select(range(max_samples))
        logger.info(f"Limited dataset to {max_samples} samples")

    logger.info(f"Loaded {len(dataset)} samples")
    return dataset


def generate_embeddings(model, texts: List[str], batch_size: int = 32):
    """
    Generate embeddings for a list of texts.

    Args:
        model: Sentence transformer model
        texts: List of text strings to embed
        batch_size: Batch size for processing

    Returns:
        List of embeddings as numpy arrays
    """
    logger.info(f"Generating embeddings for {len(texts)} texts with batch size {batch_size}")
    return model.encode(texts, batch_size=batch_size, show_progress_bar=True)


def create_chroma_collection(client, collection_name: str):
    """
    Create a new Chroma collection or get existing one.

    Args:
        client: ChromaDB client
        collection_name: Name of the collection

    Returns:
        ChromaDB collection
    """
    try:
        # Try to get existing collection
        collection = client.get_collection(collection_name)
        logger.info(f"Using existing collection '{collection_name}'")
    except:
        # Create new collection if it doesn't exist
        collection = client.create_collection(collection_name)
        logger.info(f"Created new collection '{collection_name}'")

    return collection


def main(args):
    # Use the same model name that will be used in the retriever
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    dataset = load_codesearchnet_dataset(split="train", max_samples=1000)

    # Access dataset elements correctly
    code_samples = dataset["code"]

    # Generate embeddings
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)

    logger.info(f"Connecting to ChromaDB at {args.db_path}")
    db_path = "../.././chroma_db"  # Default path for ChromaDB
    client = chromadb.PersistentClient(path=db_path)

    # Create or get collection
    collection_name = "codesearchnet"

    # Check if collection exists and delete it if specified
    try:
        client.delete_collection(collection_name)
        logger.info(f"Deleted existing collection '{collection_name}'")
    except Exception as e:
        logger.info(f"No existing collection to delete: {e}")

    collection = client.create_collection(
    name=collection_name,
    embedding_function=embedding_function,
    metadata={
        "description": "my first Chroma collection",
    }
)

    # Add data to ChromaDB
    logger.info(f"Adding {len(code_samples)} documents to ChromaDB collection")

    # Use batch processing to add documents
    batch_size = 100  # Smaller batch size for better reliability
    for i in range(0, len(code_samples), batch_size):
        end_idx = min(i + batch_size, len(code_samples))
        batch_codes = code_samples[i:end_idx]

        # Generate IDs for this batch
        batch_ids = [f"code_{i + j}" for j in range(len(batch_codes))]

        # Add documents with embeddings
        collection.add(documents=batch_codes, ids=batch_ids)

        logger.info(f"Added batch {i // batch_size + 1} to ChromaDB ({i} to {end_idx})")

    # Verify data was added
    count = collection.count()
    logger.info(f"Completed! Collection '{collection_name}' contains {count} documents")

    # Optional: Print a sample document to verify
    if count > 0:
        sample = collection.peek(limit=1)
        print(sample)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate embeddings and store in ChromaDB")
    # Add argument for database path
    parser.add_argument(
        "--db_path",
        type=str,
        default="./chroma_db",
        help="Path to the ChromaDB database (default: ./chroma_db)"
    )
    args = parser.parse_args()
    main(args)