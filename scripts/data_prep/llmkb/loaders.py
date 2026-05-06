from typing import Any, Dict, List
import json
import yaml
import os


# Loading diseases
def load_diseases(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "diseases" in data:
        diseases = data["diseases"]
    else:
        raise ValueError(
            "diseases.json must be a dict with a 'diseases' key."
        )

    for d in diseases:
        if "disease_id" not in d or "name" not in d:
            raise ValueError(f"Invalid disease entry: {d}")

    return diseases


def load_features(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "features" in data:
        features = data["features"]
    else:
        raise ValueError(
            "features file must be a dict with a 'features' key."
        )

    for feat in features:
        for key in ["feature_id", "name", "type"]:
            if key not in feat:
                raise ValueError(f"Feature missing '{key}': {feat}")

    return features


def load_config(path: str) -> dict:
    """Load YAML config, resolving path relative to caller's directory."""
    if not os.path.isabs(path):
        path = os.path.join(os.path.dirname(__file__), path)
    with open(path, "r") as f:
        return yaml.safe_load(f)