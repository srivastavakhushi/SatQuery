from unittest.mock import patch

import numpy as np

from app.sih_raster import align_temporal_pair
from tests.conftest import client, upload_images


def test_upload_does_not_run_sih_or_rsicrc():
    with patch("app.sih_raster.preprocess_temporal_pair") as mock_preprocess:
        with patch("app.tools.models.rsicrc.rsicrc_adapter.run_rsicrc") as mock_rsicrc:
            with patch("app.agent.adapters.llava_adapter.run_llava_vqa") as mock_geochat:
                with patch("app.agent.adapters.resnet_adapter.run_resnet_features") as mock_resnet:
                    image_ids = upload_images(2)
    assert len(image_ids) == 2
    mock_preprocess.assert_not_called()
    mock_rsicrc.assert_not_called()
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
    with patch("app.tools.models.rsicrc.preprocess_temporal_pair") as mock_preprocess:
        mock_preprocess.return_value = (
            __import__("numpy").zeros((3, 1, 1), dtype="float32"),
            __import__("numpy").zeros((3, 1, 1), dtype="float32"),
        )
        with patch("app.tools.models.rsicrc.rsicrc_adapter.run_rsicrc") as mock_rsicrc:
            mock_rsicrc.return_value = {
                "answer": "Buildings expanded along the river.",
                "model": "rsicrc",
                "confidence": 0.91,
            }
            response = client.post("/api/v1/query", json={
                "query": "What changed between these two images?",
                "image_ids": image_ids,
            })
    assert response.status_code == 200
    mock_preprocess.assert_called_once()
    mock_rsicrc.assert_called_once()


def test_align_temporal_pair_crops_to_shared_size():
    before = np.ones((4, 438, 441), dtype=np.float32)
    after = np.ones((4, 438, 439), dtype=np.float32) * 2
    aligned1, aligned2 = align_temporal_pair(before, after)
    assert aligned1.shape == aligned2.shape == (4, 438, 439)
    assert aligned1.shape != before.shape
    assert aligned2.shape == after.shape
