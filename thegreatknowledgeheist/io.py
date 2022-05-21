import json
from typing import Any, Dict, List

import yaml


def load_json(filepath: str) -> Dict[str, Any]:
    with open(filepath) as f:
        data = json.load(f)
    return data


def save_json(
    data: Any, filepath: str, indent: int = 4, sort_keys: bool = False
) -> None:
    with open(filepath, "w") as f:
        json.dump(data, f, indent=indent, sort_keys=sort_keys)


def load_yaml(filepath) -> Dict[str, Any]:
    with open(filepath, "r") as f:
        data = yaml.load(f, yaml.FullLoader)
    return data


def save_yaml(
    data: Any,
    filepath: str,
    default_flow_style: bool = False,
    sort_keys: bool = False,
) -> None:
    """Save data to a YAML file."""
    with open(filepath, "w") as f:
        yaml.dump(data, f, default_flow_style=default_flow_style, sort_keys=sort_keys)
