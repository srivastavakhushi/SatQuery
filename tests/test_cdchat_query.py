from unittest.mock import patch

import httpx

from app.config import settings
from tests.conftest import client, upload_images

API = "/api/v1/query"


def test_bi_temporal_change_calls_cdchat():
    image_ids = upload_images(2)
    with patch("app.tools.models.cd_chat.cdchat_adapter.run_cdchat") as mock_cdchat:
        mock_cdchat.return_value = {
            "answer": "New rooftops appeared in the post-event image.",
            "model": "cdchat",
            "confidence": 0.93,
        }
        response = client.post(API, json={
            "query": "What changed between these two images?",
            "image_ids": image_ids,
        })

    assert response.status_code == 200
    mock_cdchat.assert_called_once()
    args, kwargs = mock_cdchat.call_args
    assert kwargs["question"] == "What changed between these two images?"
    data = response.json()
    assert data["intent"] == "BI_TEMPORAL_CHANGE"
    assert "CDChat" in data["models_dispatched"]
    assert "ChangeDetection" in data["models_dispatched"]


def test_vqa_intent_does_not_call_cdchat(monkeypatch):
    from tests.conftest import FakeResponse, disable_model_mocks, install_fake_httpx

    disable_model_mocks(monkeypatch)
    monkeypatch.setattr(settings, "GEOCHAT_URL", "http://geochat.test")
    image_ids = upload_images(1)
    install_fake_httpx(
        monkeypatch,
        post=FakeResponse(200, {"answer": "Several buildings are visible.", "model": "geochat"}),
    )
    with patch("app.tools.models.cd_chat.cdchat_adapter.run_cdchat") as mock_cdchat:
        response = client.post(API, json={
            "query": "What is the building count in this satellite view?",
            "image_ids": image_ids,
        })

    assert response.status_code == 200
    data = response.json()
    assert data["intent"] == "VQA"
    mock_cdchat.assert_not_called()
    assert "CDChat" not in data["models_dispatched"]
    assert "VQA" in data["models_dispatched"]
    assert "GeoChat" in data["models_dispatched"]


def test_bi_temporal_missing_second_image():
    image_ids = upload_images(1)
    response = client.post(API, json={
        "query": "What changed between these two images?",
        "image_ids": image_ids,
    })
    assert response.status_code == 400
    assert "two image" in response.json()["detail"].lower()


def test_bi_temporal_missing_image_ids():
    response = client.post(API, json={
        "query": "What changed between these two images?",
        "image_ids": [],
    })
    assert response.status_code == 400


def test_invalid_image_id():
    response = client.post(API, json={
        "query": "What changed between these two images?",
        "image_ids": ["img-does-not-exist", "img-also-missing"],
    })
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_cdchat_unavailable(monkeypatch):
    image_ids = upload_images(2)
    monkeypatch.setattr(settings, "MODEL_MOCK_MODE", False)
    monkeypatch.setattr(settings, "CDCHAT_MOCK", False)
    monkeypatch.setattr(settings, "CDCHAT_URL", "http://127.0.0.1:8001")

    with patch("app.agent.adapters.remote.httpx.Client") as mock_client:
        mock_client.return_value.__enter__.return_value.post.side_effect = httpx.ConnectError("offline")
        response = client.post(API, json={
            "query": "What changed between these two images?",
            "image_ids": image_ids,
        })

    assert response.status_code == 503
    assert "unavailable" in response.json()["detail"].lower()


def test_cdchat_timeout(monkeypatch):
    image_ids = upload_images(2)
    monkeypatch.setattr(settings, "MODEL_MOCK_MODE", False)
    monkeypatch.setattr(settings, "CDCHAT_MOCK", False)
    monkeypatch.setattr(settings, "CDCHAT_URL", "http://127.0.0.1:8001")

    with patch("app.agent.adapters.remote.httpx.Client") as mock_client:
        mock_client.return_value.__enter__.return_value.post.side_effect = httpx.TimeoutException("timed out")
        response = client.post(API, json={
            "query": "What changed between these two images?",
            "image_ids": image_ids,
        })

    assert response.status_code == 504
    assert "timed out" in response.json()["detail"].lower()


def test_cdchat_result_is_fused_into_final_response():
    image_ids = upload_images(2)
    with patch("app.tools.models.cd_chat.cdchat_adapter.run_cdchat") as mock_cdchat:
        mock_cdchat.return_value = {
            "answer": "Buildings expanded along the river.",
            "model": "cdchat",
            "confidence": 0.91,
        }
        response = client.post(API, json={
            "query": "What changed between these two images?",
            "image_ids": image_ids,
        })

    assert response.status_code == 200
    data = response.json()
    assert "EvidenceFusion" in data["models_dispatched"]
    assert "Buildings expanded along the river." in data["answer"]
    assert data["fused_evidence"].get("cdchat", {}).get("answer") == "Buildings expanded along the river."
    assert data["execution_trace"]["selected_model"] == "CDChat"
    assert data["execution_trace"]["image_ids"] == image_ids
    assert data["execution_trace"]["selected_task"] == "BI_TEMPORAL_CHANGE"
    assert data["execution_trace"]["intent_confidence"] == 0.96


def test_upload_does_not_invoke_cdchat():
    with patch("app.tools.models.cd_chat.cdchat_adapter.run_cdchat") as mock_cdchat:
        with patch("app.agent.adapters.geochat_adapter.run_geochat_vqa") as mock_geochat:
            with patch("app.agent.adapters.popeye_adapter.run_popeye") as mock_popeye:
                with patch("app.agent.adapters.resnet_adapter.run_resnet_features") as mock_resnet:
                    upload_images(2)
    mock_cdchat.assert_not_called()
    mock_geochat.assert_not_called()
    mock_popeye.assert_not_called()
    mock_resnet.assert_not_called()
