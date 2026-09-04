import base64
import io
from typing import Any, Callable, List, Optional, Union
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.main import app

MINI_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)

client = TestClient(app)


class FakeResponse:
    def __init__(self, status_code: int = 200, data: Optional[dict] = None):
        self.status_code = status_code
        self._data = data if data is not None else {}

    def json(self):
        return self._data


def upload_images(count: int = 2, suffix: str = "png") -> List[str]:
    files = [
        ("files", (f"image{i}.{suffix}", io.BytesIO(MINI_PNG), "image/png"))
        for i in range(1, count + 1)
    ]
    response = client.post("/api/v1/upload", files=files)
    assert response.status_code == 201
    return [item["file_id"] for item in response.json()["files"]]


def disable_model_mocks(monkeypatch) -> None:
    from app.config import settings

    monkeypatch.setattr(settings, "MODEL_MOCK_MODE", False)
    monkeypatch.setattr(settings, "GEOCHAT_MOCK", False)
    monkeypatch.setattr(settings, "CDCHAT_MOCK", False)
    monkeypatch.setattr(settings, "POPEYE_MOCK", False)
    monkeypatch.setattr(settings, "RESNET_MOCK", False)


def install_fake_httpx(
    monkeypatch,
    post: Union[FakeResponse, Exception, Callable[..., Any], None] = None,
    get: Union[FakeResponse, Exception, Callable[..., Any], None] = None,
):
    posts: List[dict] = []
    gets: List[str] = []

    class FakeClient:
        def __init__(self, *args, **kwargs):
            self.timeout = kwargs.get("timeout")

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url, json=None):
            posts.append({"url": url, "json": json})
            return _resolve(post, url, json)

        def get(self, url):
            gets.append(url)
            return _resolve(get, url, None)

    monkeypatch.setattr("app.agent.adapters.remote.httpx.Client", FakeClient)
    return posts, gets


def _resolve(handler, url, json_payload):
    if handler is None:
        return FakeResponse(200, {"status": "ok"})
    if isinstance(handler, Exception):
        raise handler
    if callable(handler):
        return handler(url, json_payload)
    return handler


@pytest.fixture
def mock_cdchat_success():
    with patch("app.tools.models.cd_chat.cdchat_adapter.run_cdchat") as mock_cdchat:
        mock_cdchat.return_value = {
            "answer": "CDChat identified bi-temporal changes between the two scenes.",
            "model": "cdchat",
            "confidence": 0.94,
            "mock": False,
        }
        yield mock_cdchat
