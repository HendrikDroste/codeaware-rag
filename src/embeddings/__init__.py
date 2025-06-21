"""
This package provides a modular and extensible framework for using
various text embedding models from providers like HuggingFace and OpenAI.

It is designed to be compatible with both LangChain and ChromaDB.
"""
from .factory import get_embedding_provider, get_chroma_embedding_function
from .providers import SentenceTransformerProvider, HuggingFaceAutoModelProvider, OpenAIProvider, TFIDFProvider
from .interfaces import BaseEmbeddingProvider

# Expose the public API of the package
__all__ = [
    "get_embedding_provider",
    "get_chroma_embedding_function",
    "SentenceTransformerProvider",
    "HuggingFaceAutoModelProvider",
    "OpenAIProvider",
    "BaseEmbeddingProvider",
    "TFIDFProvider"
]
