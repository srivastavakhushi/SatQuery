"""
RSICRC adapter for bi-temporal change analysis.

Talks to the RSICRC Change Analysis API at RSICRC_URL. The live service
exposes GET / and POST /analyze as multipart files: before + after.
"""

from __future__ import annotations

import time
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import numpy as np
from PIL import Image

from app.agent.adapters import remote
from app.agent.adapters.images import source_to_rgb_png_bytes
from app.config import settings
from app.exceptions import ModelInferenceError

ImageInput = Union[str, Path, bytes, Image.Image, np.ndarray]

_ANSWER_KEYS = (
    "answer",
    "caption",
    "change_caption",
    "text",
    "output",
    "response",
    "result",
    "analysis",
    "description",
    "summary",
    "change_description",
)


def run_rsicrc(
    image1: ImageInput,
    image2: ImageInput,
    question: str = "",
) -> Dict[str, Any]:
    started = time.perf_counter()
    before_png, after_png = _equal_size_png_pair(
        source_to_rgb_png_bytes(image1, "before"),
        source_to_rgb_png_bytes(image2, "after"),
    )

    if remote.is_mock_enabled("rsicrc"):
        result = _mock_rsicrc_result(question)
    else:
        result = _http_analyze(before_png, after_png)

    result.setdefault("model", "rsicrc")
    result.setdefault("confidence", 0.9)
    result["elapsed_seconds"] = round(time.perf_counter() - started, 4)
    if question:
        result.setdefault("question", question)
    return result


def _equal_size_png_pair(before_png: bytes, after_png: bytes) -> Tuple[bytes, bytes]:
    before = Image.open(BytesIO(before_png)).convert("RGB")
    after = Image.open(BytesIO(after_png)).convert("RGB")
    width = min(before.size[0], after.size[0])
    height = min(before.size[1], after.size[1])
    if before.size != (width, height):
        before = _center_crop_rgb(before, width, height)
    if after.size != (width, height):
        after = _center_crop_rgb(after, width, height)
    return _png_bytes(before), _png_bytes(after)


def _center_crop_rgb(image: Image.Image, width: int, height: int) -> Image.Image:
    src_w, src_h = image.size
    left = max(0, (src_w - width) // 2)
    top = max(0, (src_h - height) // 2)
    return image.crop((left, top, left + width, top + height))


def _png_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _http_analyze(before_png: bytes, after_png: bytes) -> Dict[str, Any]:
    files = {
        "before": ("before.png", before_png, "image/png"),
        "after": ("after.png", after_png, "image/png"),
    }
    data = remote.post_multipart("rsicrc", settings.RSICRC_ANALYZE_PATH, files=files)
    answer = _extract_answer(data)
    if not answer:
        raise ModelInferenceError("RSICRC returned an unexpected payload.")
    result: Dict[str, Any] = {
        "answer": answer,
        "summary": answer,
        "model": data.get("model", "rsicrc"),
        "confidence": data.get("confidence", 0.9),
        "mock": False,
        "raw": data,
    }
    for key in ("objects", "detections", "change_mask", "mask"):
        if key in data:
            result[key] = data[key]
    return result


def _mock_rsicrc_result(question: str) -> Dict[str, Any]:
    answer = (
        "RSICRC identified bi-temporal changes between the two scenes, "
        "including new construction and vegetation loss."
    )
    return {
        "answer": answer,
        "summary": answer,
        "model": "rsicrc",
        "confidence": 0.94,
        "question": question,
        "mock": True,
        "task": "bi_temporal_change_detection",
    }


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
