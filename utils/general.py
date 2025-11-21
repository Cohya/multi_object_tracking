
import yaml
from typing import Dict

def read_yaml_file(file_path: str) -> Dict:
    """Reads a YAML file and returns its contents as a dictionary."""
    with open(file_path, 'r') as file:
        content = yaml.safe_load(file)
    return content