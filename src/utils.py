import os
import yaml

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