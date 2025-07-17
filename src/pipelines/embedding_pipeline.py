import os
import chromadb
from typing import List, Any, Dict, Optional
import logging
import pandas as pd

from src.utils import load_config
from src.pipelines.base_pipeline import BaseRAGPipeline
from src.embeddings.embedding_factory import get_embedding_provider
from src.embeddings.utils import create_or_get_collection
from src.embeddings.base_embedding_provider import BaseEmbeddingProvider
from src.retrievers.validate_retriever import validate_retriever, save_model_results, print_validation_results, save_results_to_csv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingPipeline(BaseRAGPipeline):
    """
    Implementation of BaseRAGPipeline with embedding models.
    This pipeline uses the same model for embedding documents and queries.
    In the presentation this implementation is referred to as "Baseline".
    """

    def __init__(
        self, 
        config_path: Optional[str] = None,
        collection_name: str = "documents",
        chroma_persist_directory: Optional[str] = None,
        reset_collection: bool = False
    ):
        """
        Initializes the EmbeddingPipeline with the specified configuration.

        Args:
            config_path: Path to the configuration file (default: config/app_config.yaml)
            collection_name: Name of the ChromaDB collection
            chroma_persist_directory: Directory for storing ChromaDB data
        """
        self.config_path = config_path or os.path.join("config", "app_config.yaml")
        self.collection_name = collection_name
        self.chroma_persist_directory = chroma_persist_directory

        # Load configuration
        self.config = load_config("app_config")
        
        # Set up embedding provider
        self.embedding_provider = self._setup_embedding_provider()
        
        # Set up ChromaDB client and collection
        self.chroma_client = None
        self.collection = None
        self.reset_collection = reset_collection
        self._setup_chroma_client()

    def _setup_embedding_provider(self) -> BaseEmbeddingProvider:
        """Sets up the embedding provider based on the configuration."""
        embedding_config = self.config["models"]["embeddings"]
        
        model_name = embedding_config["name"]
        model_vendor = embedding_config["vendor"]
        model_type = embedding_config["type"]
        
        logger.info(f"Initializing embedding model: {model_name}")
        
        return get_embedding_provider(model_name, model_vendor, model_type)

    def _setup_chroma_client(self) -> None:
        """Sets up the ChromaDB client and collection."""
        if self.chroma_persist_directory:
            self.chroma_client = chromadb.PersistentClient(self.chroma_persist_directory)
        else:
            self.chroma_client = chromadb.Client()
            
        self.collection = create_or_get_collection(
            client=self.chroma_client,
            collection_name=self.collection_name,
            embedding_function=self.embedding_provider.get_chromadb_embedding_function(),
            reset=self.reset_collection
        )

    def prepare(self, text: str) -> List[float]:
        """
        Embeds a single text.

        Args:
            text: The text to embed

        Returns:
            The embedding as a list of floats
        """
        langchain_model = self.embedding_provider.get_langchain_embedding_model()
        return langchain_model.embed_query(text)

    def prepare_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embeds a list of texts.

        Args:
            texts: The texts to embed. Can be strings or (text_for_embedding, text_for_storage) tuples.

        Returns:
            A list of embeddings
        """
        langchain_model = self.embedding_provider.get_langchain_embedding_model()
        return langchain_model.embed_documents(texts)

    def invoke(self, query: str) -> Dict[str, Any]:
        """
        Executes the complete RAG pipeline to process a query.

        Args:
            query: The input query

        Returns:
            A dictionary with relevant documents and metadata
        """
        # num_documents > 5 did not change the results by more than 0.03, so we keep it at 5 to reduce latency
        num_docs = self.config["models"]["embeddings"].get("num_documents", 5)
        
        # Execute query in the collection
        results = self.collection.query(
            query_texts=[query],
            n_results=num_docs
        )
        
        # Format result
        return {
            "query": query,
            "documents": results.get("documents", [[]]),
            "metadatas": results.get("metadatas", [[]]),
            "distances": results.get("distances", [[]]),
            "ids": results.get("ids", [[]])
        }

if __name__ == '__main__':
    # Load configuration
    config = load_config("app_config")
    collection_name = config["database"]["collection_name"]
    model_name = config["models"]["embeddings"]["name"]

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Starting embedding pipeline execution...")

    # Step 1 Create the embedding pipeline
    # Initialize pipeline for validation
    pipeline = EmbeddingPipeline(
        collection_name=collection_name,
        chroma_persist_directory="../../chroma_db",
        reset_collection=True
    )

    # Step 2: Create embeddings using embed_python_directory with pipeline parameter
    from src.embeddings.file_processor import embed_python_directory
    embed_python_directory(
        source_dir="../../../flask/src",
        pipeline=pipeline,
        chunk_size=900,
        chunk_overlap=0,
    )

    # Step 3: Validate the retriever
    logger.info("Step 2: Validating retriever...")

    # Load validation data
    validation_data = pd.read_csv('../../data/validation.csv')
    # Run validation
    results = validate_retriever(pipeline, validation_data)
    save_model_results(results,model_name)
    #print_validation_results(results)
    #save_results_to_csv(results)           # uncomment to save results for each query

