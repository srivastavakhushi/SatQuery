from typing import Any, Dict, List

from app.tools.base import BaseTool
from app.tools.models import geochat_model


class GroundingTool(BaseTool):
    @property
    def name(self) -> str:
        return "Grounding"

    @property
    def description(self) -> str:
        return "Visual Grounding and Localization Tool powered by GeoChat."

    @property
    def required_inputs(self) -> List[str]:
        return ["query"]

    def execute(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        query = payload.get("query", "")
        image_ids = payload.get("image_ids", [])
        return geochat_model.ground_target(image_ids, query)
