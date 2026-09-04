from fastapi.testclient import TestClient

from app.main import app
from tests.conftest import upload_images

client = TestClient(app)


def test_report_endpoint(mock_cdchat_success):
    image_ids = upload_images(2)
    query_resp = client.post("/api/v1/query", json={
        "query": "What changed between these two images?",
        "image_ids": image_ids,
    })
    assert query_resp.status_code == 200
    trace_id = query_resp.json()["execution_trace"]["trace_id"]

    report_resp = client.post("/api/v1/report", json={
        "trace_id": trace_id,
        "title": "Custom Change Detection Summary Report"
    })

    assert report_resp.status_code == 200
    report_data = report_resp.json()
    assert "report_id" in report_data
    assert report_data["title"] == "Custom Change Detection Summary Report"
    assert report_data["trace_id"] == trace_id
