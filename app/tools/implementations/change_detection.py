from typing import Any, Dict, List

from app.tools.base import BaseTool
from app.tools.models import cd_chat_model


class ChangeDetectionTool(BaseTool):
    @property
    def name(self) -> str:
        return "ChangeDetection"

    @property
    def description(self) -> str:
        return "Bi-Temporal Change Detection Tool powered by CDChat."

    @property
    def required_inputs(self) -> List[str]:
        return ["query"]

    def execute(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        query = payload.get("query", "")
        image_ids = payload.get("image_ids", [])
        return cd_chat_model.detect_changes(image_ids, query)
