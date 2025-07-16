import os
import yaml
import torch
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from transformers import AutoModelForCausalLM

def load_config(name: str) -> dict:
    """
    Load a configuration file from the specified path.

    Args:
        name (str): The name of the configuration file (without extension).

    Returns:
        dict: The loaded configuration as a dictionary.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(project_root, "config", f"{name}.yaml")
    with open(file_path) as file:
        config = yaml.safe_load(file)
    return config

def load_llm(type: str, model_name: str):
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
