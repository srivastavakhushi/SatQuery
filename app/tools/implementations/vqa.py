from typing import Dict, Any, List
from app.tools.base import BaseTool
from app.tools.models import geochat_model

class VQATool(BaseTool):
    @property
    def name(self) -> str:
        return "VQA"

    @property
    def description(self) -> str:
        return "Visual Question Answering Tool powered by GeoChat model."

    @property
    def required_inputs(self) -> List[str]:
        return ["query"]

    def execute(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        query = payload.get("query", "")
        image_ids = payload.get("image_ids", [])
        return geochat_model.answer_question(image_ids, query)
