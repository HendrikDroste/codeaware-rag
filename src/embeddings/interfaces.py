"""
Defines the abstract base classes and interfaces for embedding providers.
"""
from abc import ABC, abstractmethod
from langchain_core.embeddings import Embeddings

class BaseEmbeddingProvider(Embeddings, ABC):
    """
    Abstract base class for embedding providers.

    It defines a common interface for providing both LangChain and ChromaDB
    compatible embedding functions. It also inherits from LangChain's `Embeddings`
    class, allowing any subclass to be used as a LangChain embedding object.
    """

    @abstractmethod
    def get_langchain_embedding_model(self) -> Embeddings:
        """Returns a LangChain-compatible embedding model instance."""
        pass

    @abstractmethod
    def get_chromadb_embedding_function(self):
        """Returns a ChromaDB-compatible embedding function."""
        pass

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        LangChain Interface: Embeds a list of documents.

        By default, this method delegates the call to the specific
        LangChain embedding model provided by the concrete class.
        """
        return self.get_langchain_embedding_model().embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        """
        LangChain Interface: Embeds a single query text.

        By default, this method delegates the call to the specific
        LangChain embedding model provided by the concrete class.
        """
        return self.get_langchain_embedding_model().embed_query(text)
