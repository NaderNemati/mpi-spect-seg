# src/utils.py
import os
from typing import Dict, Any

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def load_config(path: str | None) -> Dict[str, Any]:
    """Load YAML config if provided; otherwise return empty dict."""
    if not path:
        return {}
    try:
        import yaml
    except ImportError:
        print("Warning: PyYAML not installed. Skipping config load.")
        return {}
    if not os.path.isfile(path):
        print(f"Warning: Config file not found: {path} (skipping)")
        return {}
    with open(path, "r") as f:
        try:
            return yaml.safe_load(f) or {}
        except Exception as e:
            print(f"Warning: Failed to parse YAML config: {e} (skipping)")
            return {}
