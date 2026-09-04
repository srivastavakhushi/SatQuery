"""
ResNet-50 supporting adapter for remote-sensing features / domain adaptation.

This project has no ResNet checkpoint or class list. The remote service at
RESNET_URL owns the checkpoint and output definition. This adapter does not
invent classes and is not a conversational model.
"""

from __future__ import annotations

from typing import Any, Dict

from app.agent.adapters import remote
from app.agent.adapters.images import image_id_to_base64_png
from app.config import settings
from app.exceptions import MissingImageIdError, ModelInferenceError


def run_resnet_features(image_id: str) -> Dict[str, Any]:
    if not image_id or not str(image_id).strip():
        raise MissingImageIdError("ResNet feature extraction requires an image ID.")

    image_b64 = image_id_to_base64_png(image_id)

    if remote.is_mock_enabled("resnet"):
        return {
            "model": "resnet-50",
            "feature_dim": 2048,
            "features": [0.0] * 8,
            "image_id": image_id,
            "task": "features",
            "mock": True,
        }

    payload = {
        "image": image_b64,
        "encoding": "base64",
    }
    data = remote.post_inference("resnet", settings.RESNET_FEATURES_PATH, payload)
    features = data.get("features")
    if features is None and data.get("feature_vector") is not None:
        features = data.get("feature_vector")
    if features is None and not any(key in data for key in ("feature_dim", "embedding", "domain")):
        raise ModelInferenceError("ResNet returned an unexpected payload.")

    result: Dict[str, Any] = {
        "model": data.get("model", "resnet-50"),
        "image_id": image_id,
        "task": "features",
        "mock": False,
        "raw": data,
    }
    if features is not None:
        result["features"] = features
        if isinstance(features, list):
            result["feature_dim"] = data.get("feature_dim", len(features))
    if data.get("feature_dim") is not None:
        result["feature_dim"] = data["feature_dim"]
    if data.get("domain") is not None:
        result["domain"] = data["domain"]
    if data.get("embedding") is not None:
        result["embedding"] = data["embedding"]
    return result
