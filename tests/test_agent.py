from unittest.mock import patch

from app.agent import run_agent_workflow
from app.config import settings
from tests.conftest import MINI_PNG


def test_agent_workflow_change_detection():
    image_ids = ["img-t1", "img-t2"]
    for image_id in image_ids:
        path = settings.UPLOAD_DIR / f"{image_id}_scene.png"
        path.write_bytes(MINI_PNG)

    with patch("app.tools.models.cd_chat.cdchat_adapter.run_cdchat") as mock_cdchat:
        mock_cdchat.return_value = {
            "answer": "CDChat identified bi-temporal changes between the two scenes.",
            "model": "cdchat",
            "confidence": 0.94,
            "mock": False,
        }
        query = "What changed between these two images?"
        result = run_agent_workflow(query, image_ids)

    assert result["intent"] == "BI_TEMPORAL_CHANGE"
    assert result["overall_confidence"] >= 0.90
    assert "ChangeDetection" in result["models_dispatched"]
    assert "CDChat" in result["models_dispatched"]
    assert "MetadataReader" in result["models_dispatched"]
    assert "EvidenceFusion" in result["models_dispatched"]
    assert result["total_latency"] > 0.0
    assert result["execution_trace"] is not None
    assert result["execution_trace"].selected_task == "BI_TEMPORAL_CHANGE"
    mock_cdchat.assert_called_once()
