from src.pipelines.base_pipeline import BaseRAGPipeline
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph
from src.embeddings.embedding_factory import get_embedding_provider
from src.embeddings.base_embedding_provider import BaseEmbeddingProvider
from langchain.chains.base import Chain
from langchain_core.language_models import BaseLanguageModel
from typing import List, Any, Dict, Optional
import os
import logging
from src.utils import load_config, load_llm
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline as hf_pipeline

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# The code is currently not functional, because the GraphCypherQAChain is unable to create a Cypher query based on the questions.
# This code could be used as a starting point for implementing a RAG pipeline that uses a Neo4j graph database.
# More details about this idea can be found in the README.md file of the repository.
class GraphPipeline(BaseRAGPipeline):
    """
    Implementation of BaseRAGPipeline using Neo4j graph database and GraphCypherQAChain.
    This pipeline uses graph query capabilities to answer questions based on code structure.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        neo4j_url: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "password",
        llm: Optional[BaseLanguageModel] = None
    ):
        """
        Initializes the GraphPipeline with Neo4j connection and LLM.

        Args:
            config_path: Path to the configuration file
            neo4j_url: URL to connect to Neo4j database
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
            llm: Language model to use for generating answers
        """
        self.config_path = config_path or os.path.join("config", "app_config.yaml")
        self.neo4j_url = neo4j_url
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.llm = llm

        # Load configuration
        self.config = load_config("app_config")

        # Set up embedding provider
        self.embedding_provider = self._setup_embedding_provider()

        # load llm from config if not provided
        if not self.llm:
            llm_config = self.config["models"]["chat"]
            model_type = llm_config["type"]
            model_name = llm_config["name"]

            self.llm = load_llm(model_type, model_name)

        # Connect to Neo4j and set up GraphCypherQAChain
        self.graph = self._setup_graph_connection()
        self.chain = self._setup_qa_chain()

    def _setup_embedding_provider(self) -> BaseEmbeddingProvider:
        """Sets up the embedding provider based on the configuration."""
        embedding_config = self.config["models"]["embeddings"]

        model_name = embedding_config["name"]
        model_vendor = embedding_config["vendor"]
        model_type = embedding_config["type"]

        logger.info(f"Initializing embedding model: {model_name}")

        return get_embedding_provider(model_name, model_vendor, model_type)

    def _setup_graph_connection(self) -> Neo4jGraph:
        """Sets up the connection to Neo4j database."""
        logger.info(f"Connecting to Neo4j at {self.neo4j_url}")
        return Neo4jGraph(
            url=self.neo4j_url,
            username=self.neo4j_user,
            password=self.neo4j_password
        )

    def _setup_qa_chain(self) -> Chain:
        """Sets up the GraphCypherQAChain."""
        if not self.llm:
            raise ValueError("Language model (llm) is required for GraphCypherQAChain")

        logger.info("Initializing GraphCypherQAChain")

        if hasattr(self.llm, "__class__") and "transformers" in str(self.llm.__class__) and not hasattr(self.llm, "bind"):
            logger.info(f"Wrapping Hugging Face model {self.llm.__class__.__name__} with LangChain compatible interface")
            pipe = hf_pipeline(
                "text-generation",
                model=self.llm,
                tokenizer=self.llm.config.name_or_path,
                max_length=2048,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
            self.llm = HuggingFacePipeline(pipeline=pipe)

        return GraphCypherQAChain.from_llm(
            llm=self.llm,
            graph=self.graph,
            verbose=True,
            return_intermediate_steps=True,
            return_direct=True,
            allow_dangerous_requests=True,
        )

    def prepare(self, text: str) -> List[float]:
        """
        Embeds a single text.

        Args:
            text: The text to embed

        Returns:
            The embedding as a list of floats
        """
        langchain_model = self.embedding_provider.get_langchain_embedding_model()
        return langchain_model.embed_query(text)

    def prepare_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embeds a list of texts.

        Args:
            texts: The texts to embed

        Returns:
            A list of embeddings
        """
        langchain_model = self.embedding_provider.get_langchain_embedding_model()
        return langchain_model.embed_documents(texts)

    def invoke(self, query: str) -> Dict[str, Any]:
        """
        Executes the complete RAG pipeline to process a query using the graph database.

        Args:
            query: The input query about code structure

        Returns:
            A dictionary with the query results
        """
        # Execute query using the GraphCypherQAChain
        result = self.chain.invoke({"query": query})

        # Extract the generated Cypher query from intermediate steps
        generated_cypher = ""
        if "intermediate_steps" in result and result["intermediate_steps"]:
            generated_cypher = result["intermediate_steps"][0].get("query", "")

        return {
            "query": query,
            "result": result["result"],
            "cypher": generated_cypher,
            "intermediate_steps": result.get("intermediate_steps", [])
        }
