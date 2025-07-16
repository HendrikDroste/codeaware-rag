"""
Defines the abstract base classes and interfaces for embedding providers.
"""
from abc import ABC, abstractmethod
from typing import List, Tuple, Union
from langchain_core.embeddings import Embeddings

# For embedding text, we allow either a single string or a tuple of two strings. The first string is used for embedding, the second is for storage.
EmbeddingText = Union[str, Tuple[str, str]]

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

    def _extract_embedding_text(self, text: EmbeddingText) -> str:
        """
        Extracts the text to be used for embedding from input which can be either
        a string or a tuple of (text_for_embedding, text_for_storage).

        Args:
            text: Either a string or a tuple of strings.

        Returns:
            The text to use for embedding.
        """
        if isinstance(text, tuple) and len(text) == 2:
            return text[0]
        return text

    def embed_documents(self, texts: List[EmbeddingText]) -> List[List[float]]:
        """
        LangChain Interface: Embeds a list of documents.

        Can handle both strings and tuples of (text_for_embedding, text_for_storage).
        Will extract the text_for_embedding from tuples.

        By default, this method delegates the call to the specific
        LangChain embedding model provided by the concrete class.
        """
        embedding_texts = [self._extract_embedding_text(text) for text in texts]
        return self.get_langchain_embedding_model().embed_documents(embedding_texts)

    def embed_query(self, text: EmbeddingText) -> List[float]:
        """
        LangChain Interface: Embeds a single query text.

        Can handle both a string and a tuple of (text_for_embedding, text_for_storage).
        Will extract the text_for_embedding from a tuple.

        By default, this method delegates the call to the specific
        LangChain embedding model provided by the concrete class.
        """
        embedding_text = self._extract_embedding_text(text)
        return self.get_langchain_embedding_model().embed_query(embedding_text)
