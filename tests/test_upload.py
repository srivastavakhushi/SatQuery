import io
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_upload_endpoint():
    file_content = b"fake image byte content"
    files = [
        ("files", ("image1.png", io.BytesIO(file_content), "image/png")),
        ("files", ("image2.png", io.BytesIO(file_content), "image/png"))
    ]
    
    response = client.post("/api/v1/upload", files=files)
    assert response.status_code == 201
    data = response.json()
    assert data["status"] == "success"
    assert len(data["files"]) == 2
    assert "file_id" in data["files"][0]
