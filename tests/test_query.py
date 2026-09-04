from fastapi.testclient import TestClient

from app.main import app
from tests.conftest import upload_images

client = TestClient(app)


def test_query_endpoint(mock_cdchat_success):
    image_ids = upload_images(2)
    payload = {
        "query": "What changed between these two images?",
        "image_ids": image_ids,
        "parameters": {}
    }

    response = client.post("/api/v1/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["intent"] == "BI_TEMPORAL_CHANGE"
    assert data["confidence"] >= 0.90
    assert "ChangeDetection" in data["models_dispatched"]
    assert "CDChat" in data["models_dispatched"]
    assert "execution_trace" in data
    assert data["execution_trace"]["selected_task"] == "BI_TEMPORAL_CHANGE"
    assert data["execution_trace"]["latency"] >= 0.0
    mock_cdchat_success.assert_called_once()
