"""
This package provides a modular and extensible framework for using
various text embedding models from providers like HuggingFace and
OpenAI.

It is designed to be compatible with both LangChain and ChromaDB.
"""

# Expose the public API of the package
__all__ = [
    "get_embedding_provider",
    "get_chroma_embedding_function",
    "SentenceTransformerProvider",
    "HuggingFaceAutoModelProvider",
    "OpenAIProvider",
    "BaseEmbeddingProvider",
    "TFIDFProvider",
]


def __getattr__(name):
    """
    Keep the same public exports without loading model libraries for
    file/graph tools.
    """
    from importlib import import_module

    if name not in __all__:
        raise AttributeError(name)
    module = (
        ".embedding_factory"
        if name.startswith("get_")
        else (
            ".base_embedding_provider"
            if name == "BaseEmbeddingProvider"
            else ".providers"
        )
    )
    value = getattr(import_module(module, __name__), name)
    globals()[name] = value
    return value
