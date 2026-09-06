"""
LLaVA adapter.

Talks to a remote GeoLLaVA VQA service at LLAVA_URL. The live wrapper exposes
GET /health and POST /vqa as multipart form fields: file (.npy) + question.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from app.agent.adapters import remote
from app.agent.adapters.images import image_id_to_npy_bytes
from app.config import settings
from app.exceptions import MissingImageIdError, ModelInferenceError

_ANGLE_BOX = re.compile(
    r"\{\s*<\s*(\d+)\s*>\s*<\s*(\d+)\s*>\s*<\s*(\d+)\s*>\s*<\s*(\d+)\s*>\s*\}"
)
_LIST_BOX = re.compile(
    r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]"
)


def run_llava_vqa(image_id: str, question: str) -> Dict[str, Any]:
    _require_image_id(image_id, "GeoLLaVA VQA")
    if remote.is_mock_enabled("llava"):
        return {
            "answer": f"GeoLLaVA mock VQA answer for: {question}",
            "model": "llava",
            "confidence": 0.5,
            "image_id": image_id,
            "task": "vqa",
            "mock": True,
        }

    npy_bytes = image_id_to_npy_bytes(image_id)
    data = _post_vqa(npy_bytes, question)
    answer = _first_text(data, "answer", "text", "output", "response")
    if not answer:
        raise ModelInferenceError("GeoLLaVA returned an unexpected payload.")
    return {
        "answer": answer,
        "model": data.get("model", "llava"),
        "confidence": data.get("confidence", 0.9),
        "image_id": image_id,
        "task": "vqa",
        "mock": False,
        "raw": data,
    }


def run_llava_caption(image_id: str, prompt: Optional[str] = None) -> Dict[str, Any]:
    _require_image_id(image_id, "GeoLLaVA captioning")
    caption_prompt = prompt or "Describe this remote-sensing scene."
    if remote.is_mock_enabled("llava"):
        caption = f"GeoLLaVA mock caption for image {image_id}."
        return {
            "caption": caption,
            "answer": caption,
            "model": "llava",
            "confidence": 0.5,
            "image_id": image_id,
            "prompt": caption_prompt,
            "task": "caption",
            "mock": True,
        }

    npy_bytes = image_id_to_npy_bytes(image_id)
    data = _post_vqa(npy_bytes, caption_prompt)
    caption = _first_text(data, "caption", "answer", "text", "output", "response")
    if not caption:
        raise ModelInferenceError("GeoLLaVA returned an unexpected payload.")
    return {
        "caption": caption,
        "answer": caption,
        "model": data.get("model", "llava"),
        "confidence": data.get("confidence", 0.9),
        "image_id": image_id,
        "prompt": caption_prompt,
        "task": "caption",
        "mock": False,
        "raw": data,
    }


def run_llava_grounding(image_id: str, query: str) -> Dict[str, Any]:
    _require_image_id(image_id, "GeoLLaVA grounding")
    if remote.is_mock_enabled("llava"):
        return {
            "objects": [],
            "answer": f"GeoLLaVA mock grounding for: {query}",
            "summary": f"GeoLLaVA mock grounding for: {query}",
            "model": "llava",
            "confidence": 0.5,
            "image_id": image_id,
            "query": query,
            "task": "grounding",
            "mock": True,
        }

    npy_bytes = image_id_to_npy_bytes(image_id)
    data = _post_vqa(npy_bytes, query)
    answer = _first_text(data, "answer", "text", "output", "caption", "response") or ""
    objects = extract_grounding_objects(data, query)
    summary = answer or (
        f"GeoLLaVA grounded {len(objects)} object(s)."
        if objects
        else "GeoLLaVA returned a grounding response without extractable coordinates."
    )
    result: Dict[str, Any] = {
        "objects": objects,
        "answer": answer or summary,
        "summary": summary,
        "model": data.get("model", "llava"),
        "image_id": image_id,
        "query": query,
        "task": "grounding",
        "mock": False,
        "raw": data,
    }
    if isinstance(data.get("confidence"), (int, float)):
        result["confidence"] = float(data["confidence"])
    elif objects:
        confidences = [
            item["confidence"]
            for item in objects
            if isinstance(item.get("confidence"), (int, float))
        ]
        if confidences:
            result["confidence"] = max(confidences)
    if "confidence" not in result:
        result["confidence"] = 0.9
    return result


def extract_grounding_objects(payload: Dict[str, Any], query: str) -> List[Dict[str, Any]]:
    """Parse boxes only when they are already structured or clearly present in text."""
    objects: List[Dict[str, Any]] = []
    raw_objects = payload.get("objects") or payload.get("grounded_boxes")
    if isinstance(raw_objects, list):
        for item in raw_objects:
            parsed = _normalize_object(item, query)
            if parsed is not None:
                objects.append(parsed)
        if objects:
            return objects

    text = _first_text(payload, "answer", "text", "output", "caption", "response") or ""
    for match in _ANGLE_BOX.finditer(text):
        objects.append(
            {
                "label": query,
                "bbox": [int(match.group(i)) for i in range(1, 5)],
            }
        )
    if objects:
        return objects
    for match in _LIST_BOX.finditer(text):
        objects.append(
            {
                "label": query,
                "bbox": [float(match.group(i)) for i in range(1, 5)],
            }
        )
    return objects


def _post_vqa(npy_bytes: bytes, question: str) -> Dict[str, Any]:
    files = {"file": ("image.npy", npy_bytes, "application/octet-stream")}
    form = {"question": question}
    return remote.post_multipart("llava", settings.LLAVA_VQA_PATH, files=files, data=form)


def _normalize_object(item: Any, query: str) -> Optional[Dict[str, Any]]:
    if not isinstance(item, dict):
        return None
    bbox = item.get("bbox") or item.get("bbox_ymin_xmin_ymax_xmax")
    polygon = item.get("polygon")
    if bbox is None and polygon is None:
        return None
    parsed: Dict[str, Any] = {
        "label": item.get("label") or query,
    }
    if bbox is not None:
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            return None
        parsed["bbox"] = [float(v) for v in bbox[:4]]
    if isinstance(polygon, (list, tuple)):
        parsed["polygon"] = polygon
    if isinstance(item.get("confidence"), (int, float)):
        parsed["confidence"] = float(item["confidence"])
    return parsed


def _first_text(data: Dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _require_image_id(image_id: Optional[str], action: str) -> None:
    if not image_id or not str(image_id).strip():
        raise MissingImageIdError(f"{action} requires an image ID.")
