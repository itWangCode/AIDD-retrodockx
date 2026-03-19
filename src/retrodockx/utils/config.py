"""YAML config loader."""
import yaml, os

def load_config(path: str = "configs/default.yaml") -> dict:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}
