import torch
import os
from transformers import AutoTokenizer, AutoModel
from chromadb.utils import embedding_functions
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings


# adapter that supports, ChromaDB and LangChain interfaces
class CustomBertEmbeddings:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.max_seq_length = 512           # maximum sequence length graphcodebert-base model
        
    def _embed(self, text):
        code_tokens = self.tokenizer.tokenize(text)
        # check if the token length exceeds the maximum sequence length
        if len(code_tokens) + 2 > self.max_seq_length:
            raise ValueError(f"Token length ({len(code_tokens) + 2}) exceeds maximum sequence length of {self.max_seq_length}. "
                            f"Please use shorter text or implement a truncation method.")
            
        tokens = [self.tokenizer.cls_token] + code_tokens + [self.tokenizer.sep_token]
        tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        tensor = torch.tensor(tokens_ids).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model(tensor)[0][0, 0].cpu().numpy()
        return embedding
    
    # ChromaDB Interface
    def __call__(self, input):
        """ChromaDB Interface: Embeds a list of texts."""
        if isinstance(input, str):
            return self._embed(input).tolist()
        return [self._embed(text).tolist() for text in input]
    
    # LangChain Interface
    def embed_query(self, text):
        """LangChain Interface: Embed a single query text."""
        return self._embed(text)
        
    def embed_documents(self, texts):
        """LangChain Interface: Embed a list of documents."""
        return [self._embed(text) for text in texts]

# ChromaDB wrapper for LangChain
class LangChainToChromaAdapter:
    def __init__(self, langchain_embedder):
        self.langchain_embedder = langchain_embedder
    
    def __call__(self, input):
        """ChromaDB Interface: Embeds a list of texts."""
        if isinstance(input, str):
            return self.langchain_embedder.embed_query(input)
        return self.langchain_embedder.embed_documents(input)

def get_embedding_function(model_name, model_type, model_vendor, for_chromadb=False):
    """
    Create an embedding function based on model parameters.
    
    Args:
        model_name: Name of the embedding model
        model_type: Type of the embedding model (e.g., "sentence-transformers")
        model_vendor: Vendor of the embedding model (e.g., "huggingface")
        for_chromadb: If True, returns a ChromaDB-compatible embedding function
        
    Returns:
        An embedding function compatible with either LangChain or ChromaDB
    """
    if str(model_vendor).lower() == "huggingface" and str(model_type).lower() == "sentence-transformers":
        if for_chromadb:
            # ChromaDB already has a special function for these models
            return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)
        else:
            return HuggingFaceEmbeddings(model_name=model_name)
            
    elif str(model_vendor).lower() == "huggingface" and str(model_type).lower() == "autotokenizer":
        # CustomBertEmbeddings supports both interfaces
        return CustomBertEmbeddings(model_name=model_name)
        
    elif str(model_vendor).lower() == "openai":
        if for_chromadb:
            # ChromaDB has a special function for OpenAI models
            try:
                return embedding_functions.OpenAIEmbeddingFunction(
                    api_key=os.environ.get("OPENAI_API_KEY"),
                    model_name=model_name
                )
            except:
                # Fallback currently not tested
                return LangChainToChromaAdapter(OpenAIEmbeddings(model=model_name))
        else:
            return OpenAIEmbeddings(model=model_name)
    else:
        raise ValueError(f"Not supported configuration: vendor={model_vendor}, type={model_type}")

def get_chroma_embedding_function(model_name, model_type, model_vendor):
    """
    Get a ChromaDB-compatible embedding function.
    
    Args:
        model_name: Name of the embedding model
        model_type: Type of the embedding model
        model_vendor: Vendor of the embedding model
    
    Returns:
        ChromaDB-compatible embedding function
    """
    return get_embedding_function(model_name, model_type, model_vendor, for_chromadb=True)
