"""
Thin FastAPI wrapper around CDChat inference.

Run separately from the SIH gateway:

    uvicorn services.cdchat.main:app --host 0.0.0.0 --port 8001
"""

from __future__ import annotations

import base64
import logging
import os
from typing import Optional

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from services.cdchat.inference import predict_change

logger = logging.getLogger(__name__)

app = FastAPI(
    title="CDChat Change Description Service",
    version="1.0.0",
    description="Inference wrapper around CDChat (batch_cdchat_vqa / load_pretrained_model).",
)


class PredictRequest(BaseModel):
    image1: str = Field(..., description="Filesystem path or base64-encoded image")
    image2: str = Field(..., description="Filesystem path or base64-encoded image")
    question: str = Field(..., description="Natural-language change question")
    encoding: str = Field("auto", description="auto | path | base64")


class PredictResponse(BaseModel):
    answer: str
    model: str = "cdchat"
    confidence: float = 0.9


def _decode_image(payload: str, encoding: str, label: str):
    mode = (encoding or "auto").lower().strip()
    if mode == "path" or (mode == "auto" and os.path.exists(payload)):
        return payload
    try:
        return base64.b64decode(payload, validate=False)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid {label} encoding.",
        ) from exc


@app.get("/health")
async def health():
    return {
        "status": "online",
        "service": "cdchat",
        "mock": os.environ.get("CDCHAT_SERVICE_MOCK", "false").lower() in {"1", "true", "yes"},
    }


@app.post("/cdchat/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    if not request.question or not request.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty.",
        )

    image1 = _decode_image(request.image1, request.encoding, "image1")
    image2 = _decode_image(request.image2, request.encoding, "image2")
    mock = os.environ.get("CDCHAT_SERVICE_MOCK", "false").lower() in {"1", "true", "yes"}

    try:
        if mock:
            answer = (
                "Mock CDChat change description: visible differences were found "
                "between the pre- and post-temporal scenes."
            )
        else:
            answer = predict_change(image1, image2, request.question.strip())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image file was not found.") from exc
    except RuntimeError as exc:
        logger.exception("CDChat runtime error")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="CDChat model is not ready. Check CDCHAT_MODEL_PATH and GPU availability.",
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("CDChat inference failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="CDChat inference failed.",
        ) from exc

    return PredictResponse(answer=answer, model="cdchat", confidence=0.9)
