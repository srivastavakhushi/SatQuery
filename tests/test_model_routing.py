from unittest.mock import patch

import httpx

from app.agent.adapters.llava_adapter import extract_grounding_objects
from app.config import settings
from tests.conftest import (
    FakeResponse,
    client,
    disable_model_mocks,
    install_fake_httpx,
    upload_images,
    upload_named_images,
)

QUERY = "/api/v1/query"
MODELS = "/api/v1/models"


def test_vqa_routes_to_llava(monkeypatch):
    disable_model_mocks(monkeypatch)
    monkeypatch.setattr(settings, "LLAVA_URL", "http://llava.test")
    image_ids = upload_images(1)
    posts, _ = install_fake_httpx(
        monkeypatch,
        post=FakeResponse(200, {"answer": "There are several warehouses.", "model": "llava"}),
    )
    response = client.post(QUERY, json={
        "query": "What is the building count in this satellite view?",
        "image_ids": image_ids,
    })
    assert response.status_code == 200
    data = response.json()
    assert data["intent"] == "VQA"
    assert "GeoLLaVA" in data["models_dispatched"]
    assert "VQA" in data["models_dispatched"]
    assert posts and "/vqa" in posts[0]["url"]
    assert posts[0]["files"] and "file" in posts[0]["files"]
    assert posts[0]["files"]["file"][0] == "image.npy"
    assert posts[0]["data"]["question"]


def test_captioning_routes_to_llava(monkeypatch):
    disable_model_mocks(monkeypatch)
    monkeypatch.setattr(settings, "LLAVA_URL", "http://llava.test")
    image_ids = upload_images(1)
    posts, _ = install_fake_httpx(
        monkeypatch,
        post=FakeResponse(200, {"caption": "A port with container ships.", "model": "llava"}),
    )
    response = client.post(QUERY, json={
        "query": "Describe the scene in detail",
        "image_ids": image_ids,
    })
    assert response.status_code == 200
    data = response.json()
    assert data["intent"] == "CAPTIONING"
    assert "GeoLLaVA" in data["models_dispatched"]
    assert "Captioning" in data["models_dispatched"]
    assert posts and "/vqa" in posts[0]["url"]
    assert posts[0]["files"] and "file" in posts[0]["files"]


def test_grounding_routes_to_llava(monkeypatch):
    disable_model_mocks(monkeypatch)
    monkeypatch.setattr(settings, "LLAVA_URL", "http://llava.test")
    image_ids = upload_images(1)
    posts, _ = install_fake_httpx(
        monkeypatch,
        post=FakeResponse(200, {
            "answer": "Ships are at {<10><20><30><40>}",
            "model": "llava",
        }),
    )
    response = client.post(QUERY, json={
        "query": "Detect and locate all cargo ships in the harbor",
        "image_ids": image_ids,
    })
    assert response.status_code == 200
    data = response.json()
    assert data["intent"] == "GROUNDING"
    assert "GeoLLaVA" in data["models_dispatched"]
    assert "Grounding" in data["models_dispatched"]
    assert "Popeye" not in data["models_dispatched"]
    assert posts and "/vqa" in posts[0]["url"]


def test_bi_temporal_routes_to_rsicrc(monkeypatch):
    disable_model_mocks(monkeypatch)
    monkeypatch.setattr(settings, "RSICRC_URL", "http://rsicrc.test")
    image_ids = upload_images(2)
    posts, _ = install_fake_httpx(
        monkeypatch,
        post=FakeResponse(200, {"answer": "New buildings appeared.", "model": "rsicrc", "confidence": 0.91}),
    )
    response = client.post(QUERY, json={
        "query": "What changed between these two images?",
        "image_ids": image_ids,
    })
    assert response.status_code == 200
    data = response.json()
    assert data["intent"] == "BI_TEMPORAL_CHANGE"
    assert "RSICRC" in data["models_dispatched"]
    assert posts and posts[0]["url"].endswith("/analyze")
    assert posts[0]["files"] and "before" in posts[0]["files"]
    assert "after" in posts[0]["files"]
    assert posts[0]["files"]["before"][0] == "before.png"
    assert posts[0]["files"]["after"][0] == "after.png"


def test_optical_sar_routes_to_popeye(monkeypatch):
    disable_model_mocks(monkeypatch)
    monkeypatch.setattr(settings, "POPEYE_URL", "http://popeye.test")
    image_ids = upload_images(2)
    posts, _ = install_fake_httpx(
        monkeypatch,
        post=FakeResponse(200, {"answer": "Optical and SAR agree on shoreline vessels.", "model": "popeye"}),
    )
    response = client.post(QUERY, json={
        "query": "Process synthetic aperture radar SAR image for cloud penetration",
        "image_ids": image_ids,
    })
    assert response.status_code == 200
    data = response.json()
    assert data["intent"] == "OPTICAL_SAR"
    assert "Popeye" in data["models_dispatched"]
    assert "OpticalSAR" in data["models_dispatched"]
    assert posts and "/analyze" in posts[0]["url"]
    assert posts[0]["files"] and "optical_image" in posts[0]["files"]
    assert "sar_image" in posts[0]["files"]
    assert posts[0]["data"]["query"]


def test_s1_s2_pair_with_generic_prompt_routes_to_popeye(monkeypatch):
    disable_model_mocks(monkeypatch)
    monkeypatch.setattr(settings, "POPEYE_URL", "http://popeye.test")
    image_ids = upload_named_images(
        ["ROIs1970_fall_s2_13_p265.png", "ROIs1970_fall_s1_13_p265.png"]
    )
    posts, _ = install_fake_httpx(
        monkeypatch,
        post=FakeResponse(200, {"answer": "Forest cover is visible in both scenes.", "model": "popeye"}),
    )
    with patch("app.sih_raster.preprocess_temporal_pair") as mock_preprocess:
        response = client.post(QUERY, json={
            "query": "forest",
            "image_ids": image_ids,
        })
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["intent"] == "OPTICAL_SAR"
    assert "Popeye" in data["models_dispatched"]
    mock_preprocess.assert_not_called()
    assert posts and "/analyze" in posts[0]["url"]


def test_facade_popeye_optical_sar(monkeypatch):
    disable_model_mocks(monkeypatch)
    monkeypatch.setattr(settings, "POPEYE_URL", "https://example.ngrok-free.dev/docs")
    image_ids = upload_images(2)
    posts, _ = install_fake_httpx(
        monkeypatch,
        post=FakeResponse(200, {"answer": "Shoreline vessels in both modalities.", "model": "popeye"}),
    )
    response = client.post(
        f"{MODELS}/popeye/optical-sar",
        json={
            "optical_image_id": image_ids[0],
            "sar_image_id": image_ids[1],
            "question": "Compare optical and SAR shoreline activity.",
        },
    )
    assert response.status_code == 200
    assert response.json()["answer"] == "Shoreline vessels in both modalities."
    assert response.json()["mock"] is False
    assert posts[0]["url"] == "https://example.ngrok-free.dev/analyze"
    assert posts[0]["files"]["optical_image"][0] == "optical.png"
    assert posts[0]["files"]["sar_image"][0] == "sar.png"
    assert posts[0]["data"]["query"] == "Compare optical and SAR shoreline activity."


def test_resnet_adapter_calls_remote_service(monkeypatch):
    disable_model_mocks(monkeypatch)
    monkeypatch.setattr(settings, "RESNET_URL", "http://resnet.test")
    image_ids = upload_images(1)
    posts, _ = install_fake_httpx(
        monkeypatch,
        post=FakeResponse(200, {"features": [0.1, 0.2, 0.3], "feature_dim": 3, "model": "resnet-50"}),
    )
    response = client.post(f"{MODELS}/resnet/features", json={"image_id": image_ids[0]})
    assert response.status_code == 200
    body = response.json()
    assert body["mock"] is False
    assert body["features"] == [0.1, 0.2, 0.3]
    assert posts and "/features" in posts[0]["url"]


def test_missing_llava_url_returns_503(monkeypatch):
    disable_model_mocks(monkeypatch)
    monkeypatch.setattr(settings, "LLAVA_URL", "")
    image_ids = upload_images(1)
    response = client.post(QUERY, json={
        "query": "What is the building count in this satellite view?",
        "image_ids": image_ids,
    })
    assert response.status_code == 503
    detail = response.json()["detail"]
    assert "GeoLLaVA inference endpoint is not configured" in detail
    assert response.json()["error"] == detail


def test_missing_rsicrc_url_returns_503(monkeypatch):
    disable_model_mocks(monkeypatch)
    monkeypatch.setattr(settings, "RSICRC_URL", "")
    image_ids = upload_images(2)
    response = client.post(QUERY, json={
        "query": "What changed between these two images?",
        "image_ids": image_ids,
    })
    assert response.status_code == 503
    assert "RSICRC inference endpoint is not configured" in response.json()["detail"]


def test_unreachable_url_returns_503(monkeypatch):
    disable_model_mocks(monkeypatch)
    monkeypatch.setattr(settings, "LLAVA_URL", "http://llava.test")
    image_ids = upload_images(1)
    install_fake_httpx(monkeypatch, post=httpx.ConnectError("offline"))
    response = client.post(QUERY, json={
        "query": "What is the building count in this satellite view?",
        "image_ids": image_ids,
    })
    assert response.status_code == 503
    assert "unavailable" in response.json()["detail"].lower()


def test_timeout_returns_504(monkeypatch):
    disable_model_mocks(monkeypatch)
    monkeypatch.setattr(settings, "LLAVA_URL", "http://llava.test")
    image_ids = upload_images(1)
    install_fake_httpx(monkeypatch, post=httpx.TimeoutException("timed out"))
    response = client.post(QUERY, json={
        "query": "What is the building count in this satellite view?",
        "image_ids": image_ids,
    })
    assert response.status_code == 504
    assert "timed out" in response.json()["detail"].lower()


def test_explicit_mock_mode_returns_mock_true(monkeypatch):
    monkeypatch.setattr(settings, "MODEL_MOCK_MODE", True)
    monkeypatch.setattr(settings, "LLAVA_URL", "")
    image_ids = upload_images(1)
    with patch("app.agent.adapters.remote.post_multipart") as mock_post:
        response = client.post(QUERY, json={
            "query": "What is the building count in this satellite view?",
            "image_ids": image_ids,
        })
    assert response.status_code == 200
    mock_post.assert_not_called()
    data = response.json()
    assert data["intent"] == "VQA"
    facade = client.post(
        f"{MODELS}/llava/vqa",
        json={"image_id": image_ids[0], "question": "What is visible?"},
    )
    assert facade.status_code == 200
    assert facade.json()["mock"] is True
    assert "mock" in facade.json()


def test_mock_responses_contain_mock_true(monkeypatch):
    monkeypatch.setattr(settings, "MODEL_MOCK_MODE", True)
    image_ids = upload_images(2)
    vqa = client.post(f"{MODELS}/llava/vqa", json={"image_id": image_ids[0], "question": "What?"})
    caption = client.post(f"{MODELS}/llava/caption", json={"image_id": image_ids[0]})
    grounding = client.post(f"{MODELS}/llava/grounding", json={"image_id": image_ids[0], "query": "ship"})
    change = client.post(
        f"{MODELS}/rsicrc/change",
        json={"image_id_1": image_ids[0], "image_id_2": image_ids[1], "question": "What changed?"},
    )
    popeye = client.post(
        f"{MODELS}/popeye/optical-sar",
        json={"optical_image_id": image_ids[0], "sar_image_id": image_ids[1], "question": "SAR view?"},
    )
    resnet = client.post(f"{MODELS}/resnet/features", json={"image_id": image_ids[0]})
    for response in (vqa, caption, grounding, change, popeye, resnet):
        assert response.status_code == 200, response.text
        assert response.json()["mock"] is True
    assert grounding.json()["objects"] == []


def test_real_mode_attempts_http_request(monkeypatch):
    disable_model_mocks(monkeypatch)
    monkeypatch.setattr(settings, "LLAVA_URL", "http://llava.test")
    image_ids = upload_images(1)
    posts, _ = install_fake_httpx(
        monkeypatch,
        post=FakeResponse(200, {"answer": "real remote answer", "model": "llava"}),
    )
    response = client.post(f"{MODELS}/llava/vqa", json={
        "image_id": image_ids[0],
        "question": "What is visible?",
    })
    assert response.status_code == 200
    assert response.json()["mock"] is False
    assert response.json()["answer"] == "real remote answer"
    assert len(posts) == 1
    assert posts[0]["url"] == "http://llava.test/vqa"
    assert posts[0]["files"] and "file" in posts[0]["files"]
    assert posts[0]["files"]["file"][0] == "image.npy"
    assert posts[0]["data"]["question"] == "What is visible?"


def test_upload_never_invokes_a_model():
    with patch("app.agent.adapters.llava_adapter.run_llava_vqa") as llava_vqa:
        with patch("app.agent.adapters.llava_adapter.run_llava_caption") as llava_cap:
            with patch("app.agent.adapters.llava_adapter.run_llava_grounding") as llava_g:
                with patch("app.agent.adapters.rsicrc_adapter.run_rsicrc") as rsicrc:
                    with patch("app.agent.adapters.popeye_adapter.run_popeye") as popeye:
                        with patch("app.agent.adapters.resnet_adapter.run_resnet_features") as resnet:
                            ids = upload_images(2)
    assert len(ids) == 2
    llava_vqa.assert_not_called()
    llava_cap.assert_not_called()
    llava_g.assert_not_called()
    rsicrc.assert_not_called()
    popeye.assert_not_called()
    resnet.assert_not_called()


def test_image_id_resolves_for_model_facade(monkeypatch):
    disable_model_mocks(monkeypatch)
    monkeypatch.setattr(settings, "LLAVA_URL", "http://llava.test")
    install_fake_httpx(monkeypatch, post=FakeResponse(200, {"answer": "ok"}))
    response = client.post(f"{MODELS}/llava/vqa", json={
        "image_id": "img-does-not-exist",
        "question": "What?",
    })
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_changedetection_does_not_call_resnet(monkeypatch):
    disable_model_mocks(monkeypatch)
    monkeypatch.setattr(settings, "RSICRC_URL", "http://rsicrc.test")
    image_ids = upload_images(2)
    install_fake_httpx(
        monkeypatch,
        post=FakeResponse(200, {"answer": "Change detected.", "model": "rsicrc", "confidence": 0.9}),
    )
    with patch("app.agent.adapters.resnet_adapter.run_resnet_features") as mock_resnet:
        response = client.post(QUERY, json={
            "query": "What changed between these two images?",
            "image_ids": image_ids,
        })
    assert response.status_code == 200
    mock_resnet.assert_not_called()


def test_models_health_does_not_expose_urls(monkeypatch):
    disable_model_mocks(monkeypatch)
    monkeypatch.setattr(settings, "LLAVA_URL", "http://secret-host/llava")
    monkeypatch.setattr(settings, "RSICRC_URL", "")
    monkeypatch.setattr(settings, "POPEYE_URL", "")
    monkeypatch.setattr(settings, "RESNET_URL", "")
    install_fake_httpx(monkeypatch, get=FakeResponse(200, {"status": "ok"}))
    response = client.get(f"{MODELS}/health")
    assert response.status_code == 200
    body = response.json()
    dumped = str(body)
    assert "secret-host" not in dumped
    assert body["llava"]["configured"] is True
    assert body["llava"]["mode"] == "remote"
    assert body["rsicrc"]["configured"] is False
    assert body["rsicrc"]["mode"] == "not_configured"


def test_health_mock_mode(monkeypatch):
    monkeypatch.setattr(settings, "MODEL_MOCK_MODE", True)
    monkeypatch.setattr(settings, "LLAVA_URL", "")
    response = client.get(f"{MODELS}/health")
    assert response.status_code == 200
    assert response.json()["llava"]["mode"] == "mock"


def test_grounding_parses_coordinates_only_when_present():
    parsed = extract_grounding_objects({"answer": "located at {<12><24><48><96>}"}, "ship")
    assert parsed == [{"label": "ship", "bbox": [12, 24, 48, 96]}]
    empty = extract_grounding_objects({"answer": "I can see several ships near the pier."}, "ship")
    assert empty == []


def test_facade_llava_vqa(monkeypatch):
    disable_model_mocks(monkeypatch)
    monkeypatch.setattr(settings, "LLAVA_URL", "http://llava.test")
    image_ids = upload_images(1)
    install_fake_httpx(monkeypatch, post=FakeResponse(200, {"answer": "a runway"}))
    response = client.post(f"{MODELS}/llava/vqa", json={
        "image_id": image_ids[0],
        "question": "What is this?",
    })
    assert response.status_code == 200
    assert response.json()["answer"] == "a runway"
    assert response.json()["mock"] is False
