"""Load the configured embedding model only when vectors are needed."""

from src.utils import load_config


def get_embedding_model(config):
    settings = load_config(path=str(config))["models"]["embeddings"]
    if (settings["vendor"], settings["type"]) != (
        "huggingface",
        "sentence-transformers",
    ):
        raise ValueError(
            "Use a Hugging Face sentence-transformers embedding model"
        )
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=settings["name"],
        model_kwargs={"trust_remote_code": True},
    )
