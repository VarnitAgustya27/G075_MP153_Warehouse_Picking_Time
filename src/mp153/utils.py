import hashlib
from pathlib import Path
from typing import Any, Dict
import yaml


def load_params(params_path: str = "params.yaml") -> Dict[str, Any]:
    """Load configuration parameters using yaml.safe_load."""
    with open(params_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_sha256(file_path: str) -> str:
    """Compute SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()
