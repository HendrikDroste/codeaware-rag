import os
import yaml
import chromadb
from chromadb.utils import embedding_functions

class ChromaSetup:
    def __init__(self):
        config = self.load_app_config("config/app_config.yaml")
        model_name = config["models"]["embeddings"]["model_name"]
        embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_name,
        )

        # Initialize the Chroma client with custom embedding function
        self.client = chromadb.Client(
            chromadb.config.Settings(
                persist_directory="./chroma_db",
            )
        )

        vectorstore = self.client.get_or_create_collection(
            name="codeaware",
            embedding_function=embedding_function,
        )



    def load_app_config(self, file_path: str) -> dict:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        file_path = os.path.join(project_root, file_path)
        with open(file_path) as file:
            config = yaml.safe_load(file)
        return config

    def get_vectorstore(self):
        return self.vectorstore