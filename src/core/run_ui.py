import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import streamlit as st
import torch
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


from src.utils import load_config
# Run via streamlit run src/core/run_ui.py

st.set_page_config(page_title="Codeaware RAG Gui", page_icon=":robot_face:", layout="wide")
st.title("Codeaware RAG Gui")
st.chat_message('ai').write("Please insert your question about the code in the input field below")

def load_model(type: str, model_name: str):
    """
    Load the model for the specified type.
    :param type: The type of model (e.g., "huggingface", "openai", "gemini").
    :param model_name: Name of the model to load.
    :return: The loaded model
    """
    if type == "huggingface":
        # Load tokenizer and model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
    elif type == "openai":
        # check if openai api key is set
        if "OPENAI_API_KEY" not in os.environ:
            raise ValueError("OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable.")
        model = ChatOpenAI(model=model_name)
    elif type == "gemini":
        # check if google api key is set
        if "GOOGLE_API_KEY" not in os.environ:
            raise ValueError("Google API key is not set. Please set the GOOGLE_API_KEY environment variable.")
        model = ChatGoogleGenerativeAI(model=model_name)
    else:
        raise ValueError(f"Model type '{type}' is not supported.")

    return model

def load_tokenizer(type: str, tokenizer_name: str):
    """
    Load the tokenizer for the specified model type.
    :param type: The type of model (e.g., "huggingface", "openai", "gemini").
    :param tokenizer_name: Name of the tokenizer to load.
    :return: The loaded tokenizer.
    """
    if type == "huggingface":
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    elif type == "openai":
        tokenizer = None  # OpenAI models do not require a separate tokenizer
    elif type == "gemini":
        tokenizer = None  # Gemini models do not require a separate tokenizer
    else:
        raise ValueError(f"Tokenizer type '{type}' is not supported.")

    return tokenizer

def load_pipeline():
    """
    Load the model pipeline for text generation.
    :return: HuggingFacePipeline for text generation.
    """
    # Load the model configuration
    config = load_config("app_config")
    model_type = config["models"]["chat"]["type"]
    model_name = config["models"]["chat"]["name"]
    tokenizer_name = config["models"]["chat"]["tokenizer"]

    tokenizer = load_tokenizer(model_type, tokenizer_name)
    model = load_model(model_type, model_name)
    # Create pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=2048,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.15
    )
    # Create LangChain wrapper for the HF pipeline
    llm = HuggingFacePipeline(pipeline=pipe)

    return llm


def query_llm(model, retriever, query):
    """
    Query the LLM with the given query and retriever.
    :param model: LLM model to use for querying.
    :param retriever: Retriever to fetch relevant documents.
    :param query: Query string to ask the LLM.
    :return: Answer provided by the LLM using the retrieved documents.
    """
    qa_chain = RetrievalQA.from_chain_type(
        llm=model,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )
    result = qa_chain.invoke({"query": query})
    print(result)
    answer = result["result"]
    # print(answer)
    st.session_state.messages.append((query, answer))
    return answer


def load_retriever():
    """
    Load the retriever for the ChromaDB collection.
    :return: VectorStoreRetriever for the ChromaDB collection.
    """

    # Load the model configuration
    config = load_config("app_config")
    model_type = config["models"]["embeddings"]["type"]
    # Use the exact same model name as used for embedding creation
    embedding_function = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}  # Ensures consistent normalization
    )

    # Create ChromaDB client with the same path
    db = Chroma(
        persist_directory="./chroma_db",
        collection_name="codesearchnet",
        embedding_function=embedding_function
    )

    # Display document count
    collection = db._collection
    document_count = collection.count()
    st.write(f"Number of documents in the database: {document_count}")

    # Optional: Display a sample to verify content
    if document_count > 0:
        sample = collection.peek(limit=1)
        st.write("Sample document metadata:", sample['metadatas'][0] if sample['metadatas'] else "No metadata")

    # Create the retriever
    retriever = db.as_retriever(search_kwargs={"k": 5})
    return retriever


if __name__ == '__main__':
    model = load_pipeline()
    if "retriever" not in st.session_state:
        st.session_state.retriever = load_retriever()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        st.chat_message('human').write(message[0])
        st.chat_message('ai').write(message[1])
    if query := st.chat_input(placeholder="Please insert your question about the code in the input field below"):
        st.chat_message("human").write(query)
        response = query_llm(model, st.session_state.retriever, query)
        st.chat_message("ai").write(response)
