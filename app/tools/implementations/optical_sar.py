from typing import Any, Dict, List

from app.tools.base import BaseTool
from app.tools.models import popeye_model


class OpticalSARTool(BaseTool):
    @property
    def name(self) -> str:
        return "OpticalSAR"

    @property
    def description(self) -> str:
        return "Optical-SAR understanding tool powered by Popeye."

    @property
    def required_inputs(self) -> List[str]:
        return []

    def execute(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        query = payload.get("query", "")
        image_ids = payload.get("image_ids", [])
        return popeye_model.process_optical_sar(image_ids, query)
