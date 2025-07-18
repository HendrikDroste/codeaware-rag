import os
from typing import List, Any, Optional
import logging
from datetime import datetime
import pandas as pd

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOpenAI
from sympy.physics.units import temperature
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms import HuggingFacePipeline

from src.utils import load_config
from src.pipelines.embedding_pipeline import EmbeddingPipeline
from src.embeddings.utils import add_documents_to_collection
from src.retrievers.validate_retriever import validate_retriever, save_model_results, print_validation_results, save_results_to_csv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingSummaryPipeline(EmbeddingPipeline):
    """
    Implementation of EmbeddingPipeline with summarization.
    This pipeline uses an LLM to summarize texts before embedding them.
    The summaries are stored in a CSV file for later usage when comparing multiple embedding models.
    """

    def __init__(
            self,
            config_path: Optional[str] = None,
            collection_name: str = "documents",
            chroma_persist_directory: Optional[str] = None,
            reset_collection: bool = False
    ):
        """
        Initializes the EmbeddingSummaryPipeline with the specified configuration.

        Args:
            config_path: Path to the configuration file (default: config/app_config.yaml)
            collection_name: Name of the ChromaDB collection
            chroma_persist_directory: Directory for storing ChromaDB data
            reset_collection: Whether to reset the existing collection
        """
        # Initialize parent class
        super().__init__(config_path, collection_name, chroma_persist_directory, reset_collection)

        # Counter for summary IDs
        self.summary_counter = 1
        self.embedding_counter = 0

        # Directory for summaries
        self.summaries_dir = os.path.join("summaries")
        os.makedirs(self.summaries_dir, exist_ok=True)

        # CSV file for summaries
        llm_config = self.config["models"]["chat"]
        model_name = llm_config.get("name", "meta-llama/Llama-3.2-1B")
        file_name = model_name.replace("/", "_").replace(":", "_") + "_summaries.csv"
        self.csv_file = os.path.join(self.summaries_dir, file_name)

        # Initialize DataFrame for summaries
        if os.path.exists(self.csv_file):
            # If the CSV file exists, we load it
            self.summaries_df = pd.read_csv(self.csv_file)
            # Set the counter to the next value after the highest existing one
            if not self.summaries_df.empty:
                self.summary_counter = self.summaries_df['id'].max() + 1
        else:
            # Otherwise create a new DataFrame
            self.summaries_df = pd.DataFrame(columns=['id', 'timestamp', 'summary'])

        # Set up LLM for summarization
        self._setup_llm()

        # Create summarization chain
        self._setup_summarization_chain()

    def _setup_llm(self) -> None:
        """Sets up the LLM for summarization."""
        llm_config = self.config["models"]["chat"]
        model_name = llm_config.get("name", "meta-llama/Llama-3.2-1B")
        model_vendor = llm_config.get("vendor", "huggingface")

        logger.info(f"Initializing LLM for summarization: {model_name} from {model_vendor}")

        if model_vendor.lower() == "openai":
            self.llm = ChatOpenAI(model_name=model_name, temperature=0.0)
        elif model_vendor.lower() == "huggingface":
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)

            text_gen_pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                do_sample=False,
                max_new_tokens=256,
                pad_token_id=tokenizer.eos_token_id,
                return_full_text=False,
                temperature=None,
                top_p=None,
            )

            self.llm = HuggingFacePipeline(pipeline=text_gen_pipeline)
        else:
            raise ValueError(f"Unsupported LLM vendor: {model_vendor}")

    def _setup_summarization_chain(self) -> None:
        """Sets up the chain for text summarization."""
        template = """Imagine you are a software engineer tasked with summarizing the following Code. Explain the main points clearly and concisely, focusing on the most important aspects.
        You will be provided with code snippets, comments, and other technical details. Your goal is to create a summary that captures the essence of the text while being understandable to someone who may not be familiar with the specific code or context.
        Do not repeat yourself!
        
        Code:
        {text}
        
        Summary:
        """

        prompt = PromptTemplate.from_template(template)
        self.summarization_chain = prompt | self.llm | StrOutputParser()

    def _save_summary_to_file(self, summary: str) -> int:
        """
        Saves a summary in a pandas DataFrame and updates the CSV file.

        Args:
            summary: The summary to be saved

        Returns:
            The ID of the saved summary
        """
        current_id = self.summary_counter
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Add new row to DataFrame
        new_row = pd.DataFrame({
            'id': [current_id],
            'timestamp': [timestamp],
            'summary': [summary]
        })

        # Update DataFrame
        self.summaries_df = pd.concat([self.summaries_df, new_row], ignore_index=True)

        # Save DataFrame to CSV file
        self.summaries_df.to_csv(self.csv_file, index=False)

        logger.info(f"Summary with ID {current_id} saved in DataFrame and CSV updated")
        self.summary_counter += 1
        return current_id

    def _summarize_text(self, text: str) -> str:
        """
        Generates a summary of the input text using the LLM.

        Args:
            text: Text to summarize

        Returns:
            Summarized text
        """
        try:
            logger.info(f"Generating summary for text of length: {len(text)}")
            summary = self.summarization_chain.invoke({"text": text})

            if not summary or summary.strip() == "":
                logger.warning("Empty summary generated, using fallback")
                summary = text[:200] + "..." if len(text) > 200 else text
            elif "Summary:" in summary:
                # Extract only the part after "Summary:"
                summary = summary.split("Summary:")[-1].strip()

            logger.info(f"Summary successfully generated ({len(summary)} characters)")
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            # Fallback summary
            summary = text[:200] + "..." if len(text) > 200 else text

        # Save summary to CSV file
        self._save_summary_to_file(summary)

        return summary

    def prepare(self, text: str) -> List[float]:
        """
        Summarizes and creates the embedding for a single text.

        Args:
            text: The text to embed

        Returns:
            The embedding as a list of floats
        """
        summary = self._summarize_text(text)
        langchain_model = self.embedding_provider.get_langchain_embedding_model()
        return langchain_model.embed_query(summary)

    def prepare_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Summarizes and embeds a list of texts.

        Args:
            texts: The texts to embed

        Returns:
            A list of embeddings
        """
        # Check if summaries already exist in the CSV file.
        # We therefore ensure that we use the same summaries for all embedding models.
        # This ensures that all models are compared on the same summaries.
        if os.path.exists(self.csv_file) and self.embedding_counter+len(texts) <= self.summaries_df.shape[0]:
            logger.info(f"Using existing summaries from {self.csv_file}")
            summaries = self.summaries_df['summary'].tolist()
            summaries = summaries[self.embedding_counter:self.embedding_counter+len(texts)]
            self.embedding_counter += len(summaries)
        else:
            summaries = [self._summarize_text(text) for text in texts]
            self.embedding_counter += len(summaries)

        langchain_model = self.embedding_provider.get_langchain_embedding_model()
        return langchain_model.embed_documents(summaries)

if __name__ == '__main__':
    # Load configuration
    config = load_config("app_config")
    collection_name = config["database"]["collection_name"]
    model_name = config["models"]["embeddings"]["name"]

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Starting embedding pipeline execution...")

    # Step 1 Create the embedding pipeline
    # Initialize pipeline for validation
    pipeline =  EmbeddingSummaryPipeline(
        collection_name=collection_name,
        chroma_persist_directory="../../chroma_db",
        reset_collection=True
    )

    # Step 2: Create embeddings using embed_python_directory with pipeline parameter
    from src.embeddings.file_processor import embed_python_directory
    embed_python_directory(
        source_dir="../../../flask/src",
        pipeline=pipeline,
        chunk_size=900,
        chunk_overlap=0,
    )

    # Step 3: Validate the retriever
    logger.info("Step 2: Validating retriever...")

    # Load validation data
    validation_data = pd.read_csv('../../data/validation.csv')

    # Run validation
    results = validate_retriever(pipeline, validation_data)
    save_model_results(results, model_name)
    #print_validation_results(results)
    #save_results_to_csv(results)