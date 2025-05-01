import os
import streamlit as st
import yaml
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
# Run via streamlit run src/core/run_ui.py

st.set_page_config(page_title="Codeaware RAG Gui", page_icon=":robot_face:", layout="wide")
st.title("Codeaware RAG Gui")
st.chat_message('ai').write("Please insert your question about the code in the input field below")


def load_app_config(file_path: str) -> dict:
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    file_path = os.path.join(project_root, file_path)
    with open(file_path) as file:
        config = yaml.safe_load(file)
    return config


def load_llm():
    from langchain_huggingface import HuggingFacePipeline
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    import torch

    # Choose a model that fits your needs and hardware capabilities
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Example model

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )

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
    qa_chain = RetrievalQA.from_chain_type(
        llm=model,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )
    result = qa_chain.invoke({"query": query})
    print(result)
    answer = result["result"]
    #print(answer)
    st.session_state.messages.append((query, answer))
    return answer


def load_retriever():
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
    model = load_llm()
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