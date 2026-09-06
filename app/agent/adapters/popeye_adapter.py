"""
Popeye adapter for optical + SAR understanding.

Talks to the SatQuery wrapper at POPEYE_URL. The live service exposes
GET /health and POST /analyze as multipart form fields:
query + optical_image + sar_image.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from app.agent.adapters import remote
from app.agent.adapters.images import image_id_to_png_bytes
from app.config import settings
from app.exceptions import MissingImageIdError, ModelInferenceError

_ANSWER_KEYS = (
    "answer",
    "text",
    "output",
    "caption",
    "response",
    "result",
    "analysis",
    "final_answer",
    "description",
    "summary",
)


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

    optical_png = image_id_to_png_bytes(optical_id) if optical_id else None
    sar_png = image_id_to_png_bytes(sar_id) if sar_id else None
    if optical_png is None:
        optical_png = sar_png
    if sar_png is None:
        sar_png = optical_png

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

    files = {
        "optical_image": ("optical.png", optical_png, "image/png"),
        "sar_image": ("sar.png", sar_png, "image/png"),
    }
    form = {"query": question}
    data = remote.post_multipart("popeye", settings.POPEYE_PREDICT_PATH, files=files, data=form)
    answer = _extract_answer(data)
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


def _extract_answer(data: Dict[str, Any]) -> str:
    text = _first_text(data, *_ANSWER_KEYS)
    if text:
        return text
    for key in _ANSWER_KEYS:
        value = data.get(key)
        if isinstance(value, dict):
            nested = _first_text(value, *_ANSWER_KEYS)
            if nested:
                return nested
    return ""


def _first_text(data: Dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""
