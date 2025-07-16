"""
Contains concrete implementations for various embedding model providers.
"""
import os
import torch
from typing import List, Union, Tuple, Any
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions, BaseModelOutputWithPooling
from transformers import AutoTokenizer, AutoModel
from chromadb.utils import embedding_functions
from sklearn.feature_extraction.text import TfidfVectorizer

# Local import from the same package
from .base_embedding_provider import BaseEmbeddingProvider, EmbeddingText


class SentenceTransformerProvider(BaseEmbeddingProvider):
    """Provider for Sentence-Transformer models from HuggingFace."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._langchain_embedder = None

    def get_langchain_embedding_model(self) -> Embeddings:
        """Returns a LangChain-compatible HuggingFaceEmbeddings model."""
        if self._langchain_embedder is None:
            self._langchain_embedder = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={"trust_remote_code": True}
            )
        return self._langchain_embedder

    def get_chromadb_embedding_function(self):
        """Returns a ChromaDB-native SentenceTransformerEmbeddingFunction."""
        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.model_name, trust_remote_code=True
        )


class HuggingFaceAutoModelProvider(BaseEmbeddingProvider):
    """
    Provider for custom HuggingFace AutoModel implementations.
    This class implements both the LangChain and ChromaDB interfaces directly.

    The SentenceTransformerProvider should be preferred over this one, as many of the models implemented here need special handling, often due to their architecture or tokenization requirements.
    Therefore, not all the pipelines will work out of the box with this provider.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.max_seq_length = getattr(self.tokenizer, 'model_max_length', 512)

    def _prepare_inputs(self, text: str):
        """Prepares the tokenized input for the model."""
        if self.model_name == "microsoft/unixcoder-base":
            tokens = self.tokenizer.tokenize(text)
            tokens = tokens[:self.max_seq_length - 4]
            tokens = [self.tokenizer.cls_token] + tokens + [self.tokenizer.sep_token]
            tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            return torch.tensor(tokens_ids).unsqueeze(0).to(self.device)
        else:
            return self.tokenizer.encode(
                text, return_tensors="pt", truncation=True, max_length=self.max_seq_length
            ).to(self.device)

    def _embed(self, text: EmbeddingText) -> List[float]:
        """Generates an embedding for a single piece of text."""
        embedding_text = self._extract_embedding_text(text)
        tensor = self._prepare_inputs(embedding_text)
        with torch.no_grad():
            result = self.model(tensor)
            if isinstance(result, (BaseModelOutputWithPoolingAndCrossAttentions, BaseModelOutputWithPooling)):
                embedding = result.pooler_output[0].cpu().numpy()
            else:
                embedding = result[0][:, 0, :].cpu().numpy().flatten()
        return embedding.tolist()

    def get_langchain_embedding_model(self) -> Embeddings:
        """Returns self as it conforms to the LangChain Embeddings interface."""
        return self

    def get_chromadb_embedding_function(self):
        """Returns a wrapper function that extracts the correct text for embedding."""
        def wrapper_function(input_texts: Union[List[EmbeddingText], EmbeddingText]) -> Union[List[List[float]], List[float]]:
            if isinstance(input_texts, (str, tuple)):
                return self._embed(input_texts)
            return [self._embed(text) for text in input_texts]
        return wrapper_function

    def embed_documents(self, texts: List[EmbeddingText]) -> List[List[float]]:
        """LangChain Interface: Embed a list of documents."""
        return [self._embed(text) for text in texts]

    def embed_query(self, text: EmbeddingText) -> List[float]:
        """LangChain Interface: Embed a single query text."""
        return self._embed(text)

    def __call__(self, input_texts: Union[List[EmbeddingText], EmbeddingText]) -> Union[List[List[float]], List[float]]:
        """ChromaDB Interface: Embeds a list of texts or a single text."""
        if isinstance(input_texts, (str, tuple)):
             return self._embed(input_texts)
        return [self._embed(text) for text in input_texts]


class OpenAIProvider(BaseEmbeddingProvider):
    """Provider for OpenAI embedding models."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("To use OpenAI embeddings, please set the OPENAI_API_KEY env var.")
        self._langchain_embedder = None
        self._chromadb_embedder = None

    def get_langchain_embedding_model(self) -> Embeddings:
        """Returns a LangChain-compatible OpenAIEmbeddings model."""
        if self._langchain_embedder is None:
            self._langchain_embedder = OpenAIEmbeddings(
                model=self.model_name, api_key=self.api_key
            )
        return self._langchain_embedder

    def get_chromadb_embedding_function(self):
        """Returns a ChromaDB-native OpenAIEmbeddingFunction wrapped to handle tuples."""
        if self._chromadb_embedder is None:
            original_embedder = embedding_functions.OpenAIEmbeddingFunction(
                api_key=self.api_key, model_name=self.model_name
            )

            # Create a wrapper that handles tuples
            def wrapper_function(input_texts):
                if isinstance(input_texts, (str, tuple)):
                    text_to_embed = self._extract_embedding_text(input_texts)
                    return original_embedder([text_to_embed])[0]
                else:
                    texts_to_embed = [self._extract_embedding_text(text) for text in input_texts]
                    return original_embedder(texts_to_embed)

            self._chromadb_embedder = wrapper_function
        return self._chromadb_embedder



# WORK IN PROGRESS: This provider is not yet fully functional.
class TFIDFProvider(BaseEmbeddingProvider):
    """
    Provider for Scikit-learn's TF-IDF vectorizer.

    Note: This provider is stateful. It fits the TF-IDF model on the first
    batch of documents it is asked to embed. For use in a retrieval scenario,
    the fitted `vectorizer` object must be persisted (e.g., using pickle) and
    loaded along with the vector database to ensure queries are vectorized with
    the same vocabulary.
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    @property
    def _is_fitted(self) -> bool:
        """Check if the vectorizer has been fitted."""
        return hasattr(self.vectorizer, 'vocabulary_')

    def embed_documents(self, texts: List[EmbeddingText]) -> List[List[float]]:
        """
        Embed a list of documents. If the model is not yet fitted, it will be
        fitted on this list of documents.
        """
        embedding_texts = [self._extract_embedding_text(text) for text in texts]

        if self._is_fitted:
            return self.vectorizer.transform(embedding_texts).toarray().tolist()
        else:
            # If not fitted, fit and then transform
            return self.vectorizer.fit_transform(embedding_texts).toarray().tolist()

    def embed_query(self, text: EmbeddingText) -> List[float]:
        """Embed a single query text."""
        if not self._is_fitted:
            raise RuntimeError(
                "TF-IDF provider is not fitted. You must call 'embed_documents' "
                "with the corpus before embedding a query."
            )
        embedding_text = self._extract_embedding_text(text)
        return self.vectorizer.transform([embedding_text]).toarray().tolist()[0]

    def get_langchain_embedding_model(self) -> Embeddings:
        """Returns self as it conforms to the LangChain Embeddings interface."""
        return self

    def get_chromadb_embedding_function(self):
        """Returns self as it is a callable that ChromaDB can use."""
        return self

    def __call__(self, input_texts: Union[List[EmbeddingText], EmbeddingText]) -> Union[List[List[float]], List[float]]:
        """ChromaDB Interface: Embeds a list of texts or a single text."""
        if isinstance(input_texts, (str, tuple)):
             return self.embed_query(input_texts)
        # When ChromaDB calls this with a list, it's for embedding documents
        return self.embed_documents(input_texts)
