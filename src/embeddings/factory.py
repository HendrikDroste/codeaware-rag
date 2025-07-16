"""
Factory function for creating and retrieving embedding providers.
"""
# Local imports from the same package
from .interfaces import BaseEmbeddingProvider
from .providers import SentenceTransformerProvider, HuggingFaceAutoModelProvider, OpenAIProvider, TFIDFProvider

def get_embedding_provider(model_name: str, model_vendor: str, model_type: str) -> BaseEmbeddingProvider:
    """
    Factory function to get the correct embedding provider instance.

    Args:
        model_name: Name of the embedding model.
        model_vendor: The vendor of the model (e.g., "huggingface", "openai").
        model_type: The type of the model (e.g., "sentence-transformers").

    Returns:
        An instance of a BaseEmbeddingProvider subclass.
    """
    vendor = str(model_vendor).lower()
    m_type = str(model_type).lower()

    if vendor == "huggingface":
        if m_type == "sentence-transformers":
            return SentenceTransformerProvider(model_name)
        elif m_type == "automodel":
            return HuggingFaceAutoModelProvider(model_name)
    elif vendor == "openai":
        return OpenAIProvider(model_name)
    elif vendor == "sklearn" and m_type == "tfidf":
        return TFIDFProvider()

    raise ValueError(f"Unsupported config: vendor='{model_vendor}', type='{m_type}'")

def get_chroma_embedding_function(model_name: str, model_type: str, model_vendor: str):
    """
    Helper to get a ChromaDB-compatible embedding function directly.
    Maintains backward compatibility with previous versions.

    Args:
        model_name: Name of the embedding model.
        model_type: The type of the model.
        model_vendor: The vendor of the model.

    Returns:
        A ChromaDB-compatible embedding function.
    """
    provider = get_embedding_provider(model_name, model_vendor, model_type)
    return provider.get_chromadb_embedding_function()
