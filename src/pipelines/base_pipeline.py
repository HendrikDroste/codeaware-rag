from abc import ABC, abstractmethod
from typing import List, Any


class BaseRAGPipeline(ABC):
    """
    Abstract base class for a Retrieval-Augmented Generation (RAG) pipeline.
    This class defines the interface for embedding and querying text data.
    """

    @abstractmethod
    def embed(self, text: str) -> None:
        """
        Embed a single string into a vector representation.

        Args:
            text (str): The input string to embed.

        Returns:
            Any: The embedded vector representation.
        """
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> None:
        """
        Embed a batch of strings.

        Args:
            texts (List[str]): The list of strings to embed.
        """
        pass

    @abstractmethod
    def invoke(self, query: str) -> str:
        """
        Run the full RAG pipeline to process a query and return a response.

        Args:
            query (str): The input query string.

        Returns:
            str: The generated response from the pipeline.
        """
        pass