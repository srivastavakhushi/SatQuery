"""
Popeye adapter for optical + SAR understanding.

No public Popeye HTTP API or in-repo implementation was found. This adapter
uses OUR provider-agnostic contract against POPEYE_URL. The remote GPU service
loads the actual Popeye weights.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from app.agent.adapters import remote
from app.agent.adapters.images import image_id_to_base64_png
from app.config import settings
from app.exceptions import MissingImageIdError, ModelInferenceError


def run_popeye(
    question: str,
    optical_image_id: Optional[str] = None,
    sar_image_id: Optional[str] = None,
) -> Dict[str, Any]:
    optical_id = _clean_id(optical_image_id)
    sar_id = _clean_id(sar_image_id)
    if not optical_id and not sar_id:
        raise MissingImageIdError(
            "Popeye optical-SAR analysis requires at least one optical or SAR image ID."
        )

    optical_b64 = image_id_to_base64_png(optical_id) if optical_id else None
    sar_b64 = image_id_to_base64_png(sar_id) if sar_id else None

    if remote.is_mock_enabled("popeye"):
        answer = f"Popeye mock optical-SAR response for: {question}"
        return {
            "answer": answer,
            "summary": answer,
            "model": "popeye",
            "confidence": 0.5,
            "optical_image_id": optical_id,
            "sar_image_id": sar_id,
            "task": "optical_sar",
            "mock": True,
        }

    payload = {
        "optical_image": optical_b64,
        "sar_image": sar_b64,
        "question": question,
        "encoding": "base64",
    }
    data = remote.post_inference("popeye", settings.POPEYE_PREDICT_PATH, payload)
    answer = _first_text(data, "answer", "text", "output", "caption")
    if not answer:
        raise ModelInferenceError("Popeye returned an unexpected payload.")
    result: Dict[str, Any] = {
        "answer": answer,
        "summary": answer,
        "model": data.get("model", "popeye"),
        "confidence": data.get("confidence", 0.9),
        "optical_image_id": optical_id,
        "sar_image_id": sar_id,
        "task": "optical_sar",
        "mock": False,
        "raw": data,
    }
    if isinstance(data.get("objects"), list):
        result["objects"] = data["objects"]
    if isinstance(data.get("detections"), list):
        result["detections"] = data["detections"]
    return result


def _clean_id(image_id: Optional[str]) -> Optional[str]:
    if image_id is None:
        return None
    cleaned = str(image_id).strip()
    return cleaned or None


def _first_text(data: Dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""
