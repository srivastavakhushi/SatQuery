from unittest.mock import patch

from tests.conftest import client, upload_images


def test_upload_does_not_run_sih_or_cdchat():
    with patch("app.sih_raster.preprocess_temporal_pair") as mock_preprocess:
        with patch("app.tools.models.cd_chat.cdchat_adapter.run_cdchat") as mock_cdchat:
            with patch("app.agent.adapters.geochat_adapter.run_geochat_vqa") as mock_geochat:
                with patch("app.agent.adapters.resnet_adapter.run_resnet_features") as mock_resnet:
                    image_ids = upload_images(2)
    assert len(image_ids) == 2
    mock_preprocess.assert_not_called()
    mock_cdchat.assert_not_called()
    mock_geochat.assert_not_called()
    mock_resnet.assert_not_called()


def test_upload_unreadable_file_still_stored():
    files = [("files", ("image1.png", b"fake image byte content", "image/png"))]
    response = client.post("/api/v1/upload", files=files)
    assert response.status_code == 201
    record = response.json()["files"][0]
    assert record["file_id"].startswith("img-")
    assert "filepath" in record


def test_query_bi_temporal_uses_sih_preprocess():
    image_ids = upload_images(2)
    with patch("app.tools.models.cd_chat.preprocess_temporal_pair") as mock_preprocess:
        mock_preprocess.return_value = (
            __import__("numpy").zeros((3, 1, 1), dtype="float32"),
            __import__("numpy").zeros((3, 1, 1), dtype="float32"),
        )
        with patch("app.tools.models.cd_chat.cdchat_adapter.run_cdchat") as mock_cdchat:
            mock_cdchat.return_value = {
                "answer": "Buildings expanded along the river.",
                "model": "cdchat",
                "confidence": 0.91,
            }
            response = client.post("/api/v1/query", json={
                "query": "What changed between these two images?",
                "image_ids": image_ids,
            })
    assert response.status_code == 200
    mock_preprocess.assert_called_once()
    mock_cdchat.assert_called_once()
