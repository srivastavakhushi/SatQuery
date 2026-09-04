"""
Provider-agnostic HTTP client for remote model inference.

This backend never loads model weights. It POSTs prepared payloads to
configured URLs. Missing URLs and network failures do not fall back to mock data.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import httpx

from app.config import settings
from app.exceptions import (
    ModelInferenceError,
    ModelNotConfiguredError,
    ModelTimeoutError,
    ModelUnavailableError,
)

_MODEL_META: Dict[str, Tuple[str, str, str, str]] = {
    "geochat": ("GEOCHAT_URL", "GEOCHAT_MOCK", "GEOCHAT_TIMEOUT_SECONDS", "GeoChat"),
    "cdchat": ("CDCHAT_URL", "CDCHAT_MOCK", "CDCHAT_TIMEOUT_SECONDS", "CDChat"),
    "popeye": ("POPEYE_URL", "POPEYE_MOCK", "POPEYE_TIMEOUT_SECONDS", "Popeye"),
    "resnet": ("RESNET_URL", "RESNET_MOCK", "RESNET_TIMEOUT_SECONDS", "ResNet"),
}


def display_name(model: str) -> str:
    return _MODEL_META[model][3]


def is_mock_enabled(model: str) -> bool:
    mock_attr = _MODEL_META[model][1]
    return bool(settings.MODEL_MOCK_MODE) or bool(getattr(settings, mock_attr))


def configured_url(model: str) -> str:
    url_attr = _MODEL_META[model][0]
    return str(getattr(settings, url_attr) or "").strip()


def require_remote_url(model: str) -> str:
    url = configured_url(model)
    if not url:
        raise ModelNotConfiguredError(
            f"{display_name(model)} inference endpoint is not configured"
        )
    return url.rstrip("/")


def join_url(base: str, path: str) -> str:
    if not path:
        return base
    if path.startswith("http://") or path.startswith("https://"):
        return path
    return f"{base.rstrip('/')}/{str(path).lstrip('/')}"


def post_inference(model: str, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    name = display_name(model)
    url = join_url(require_remote_url(model), path)
    timeout = float(getattr(settings, _MODEL_META[model][2]))
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(url, json=payload)
    except httpx.TimeoutException as exc:
        raise ModelTimeoutError(f"{name} inference timed out.") from exc
    except httpx.RequestError as exc:
        raise ModelUnavailableError(f"{name} service is unavailable.") from exc

    if response.status_code == 504:
        raise ModelTimeoutError(f"{name} inference timed out.")
    if response.status_code == 503:
        raise ModelUnavailableError(
            _response_detail(response, f"{name} service is unavailable.")
        )
    if response.status_code >= 500:
        raise ModelInferenceError(
            _response_detail(response, f"{name} inference failed.")
        )
    if response.status_code >= 400:
        raise ModelInferenceError(
            _response_detail(response, f"{name} inference failed.")
        )

    try:
        data = response.json()
    except ValueError as exc:
        raise ModelInferenceError(f"{name} returned a non-JSON response.") from exc
    if not isinstance(data, dict):
        raise ModelInferenceError(f"{name} returned an unexpected payload.")
    return data


def probe_health(model: str) -> Dict[str, Any]:
    url = configured_url(model)
    mock = is_mock_enabled(model)
    if mock:
        mode = "mock"
    elif not url:
        mode = "not_configured"
    else:
        mode = "remote"

    reachable = False
    if url:
        base = url.rstrip("/")
        timeout = float(settings.MODEL_HEALTH_TIMEOUT_SECONDS)
        try:
            with httpx.Client(timeout=timeout) as client:
                for candidate in (f"{base}/health", base):
                    try:
                        response = client.get(candidate)
                    except httpx.RequestError:
                        continue
                    if response.status_code < 500:
                        reachable = True
                        break
        except httpx.RequestError:
            reachable = False

    return {
        "configured": bool(url),
        "reachable": reachable,
        "mode": mode,
    }


def _response_detail(response: httpx.Response, fallback: str) -> str:
    try:
        body = response.json()
        if isinstance(body, dict):
            detail = body.get("detail") or body.get("error")
            if isinstance(detail, str) and detail.strip():
                return detail
    except ValueError:
        pass
    return fallback
