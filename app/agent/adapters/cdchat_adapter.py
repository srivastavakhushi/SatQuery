"""
CDChat adapter: Sih-processed arrays → CDChat RGB → CDChat HTTP service.

Raster/GeoTIFF preprocessing stays in Sih. This module only converts
Sih's (bands, H, W) float32 output into the RGB PIL/PNG that
cdchat.eval.batch_cdchat_vqa.eval_model consumes.

If CDCHAT_URL is localhost, the CDChat weights run on this machine.
The default architecture assumes a remote GPU wrapper.
"""

from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
from PIL import Image

from app.agent.adapters import remote
from app.agent.adapters.images import source_to_rgb_png_bytes
from app.config import settings
from app.exceptions import ModelInferenceError

ImageInput = Union[str, Path, bytes, Image.Image, np.ndarray]


def run_cdchat(
    image1: ImageInput,
    image2: ImageInput,
    question: str,
) -> Dict[str, Any]:
    started = time.perf_counter()
    png1 = source_to_rgb_png_bytes(image1, "image1")
    png2 = source_to_rgb_png_bytes(image2, "image2")

    if remote.is_mock_enabled("cdchat"):
        result = _mock_cdchat_result(question)
    else:
        result = _http_predict(png1, png2, question)

    result.setdefault("model", "cdchat")
    result.setdefault("confidence", 0.9)
    result["elapsed_seconds"] = round(time.perf_counter() - started, 4)
    return result


def sih_array_to_rgb(processed: np.ndarray) -> Image.Image:
    from app.agent.adapters.images import sih_array_to_rgb as _sih_array_to_rgb

    return _sih_array_to_rgb(processed)


def _http_predict(png1: bytes, png2: bytes, question: str) -> Dict[str, Any]:
    payload = {
        "image1": base64.b64encode(png1).decode("ascii"),
        "image2": base64.b64encode(png2).decode("ascii"),
        "question": question,
        "encoding": "base64",
    }
    data = remote.post_inference("cdchat", settings.CDCHAT_PREDICT_PATH, payload)
    if "answer" not in data:
        raise ModelInferenceError("CDChat returned an unexpected payload.")
    return {
        "answer": data.get("answer"),
        "model": data.get("model", "cdchat"),
        "confidence": data.get("confidence", 0.9),
        "mock": False,
        "raw": data,
    }


def _mock_cdchat_result(question: str) -> Dict[str, Any]:
    answer = (
        "CDChat identified bi-temporal changes between the two scenes, "
        "including new construction and vegetation loss."
    )
    return {
        "answer": answer,
        "model": "cdchat",
        "confidence": 0.94,
        "summary": answer,
        "question": question,
        "mock": True,
        "task": "bi_temporal_change_detection",
    }
