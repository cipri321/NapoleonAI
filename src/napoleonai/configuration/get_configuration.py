from typing import Dict

import yaml


def get_configuration(filepath: str) -> Dict:
    with open(filepath, "r") as input_file:
        return yaml.safe_load(input_file)
