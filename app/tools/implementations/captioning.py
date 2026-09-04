from typing import Any, Dict, List

from app.tools.base import BaseTool
from app.tools.models import geochat_model


class CaptioningTool(BaseTool):
    @property
    def name(self) -> str:
        return "Captioning"

    @property
    def description(self) -> str:
        return "Image Captioning Tool powered by GeoChat."

    @property
    def required_inputs(self) -> List[str]:
        return []

    def execute(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        image_ids = payload.get("image_ids", [])
        prompt = payload.get("query") or "Describe this remote-sensing scene."
        result = geochat_model.generate_caption(image_ids, prompt=prompt)
        caption_text = result.get("caption") or result.get("answer") or ""
        result["summary"] = caption_text
        result["answer"] = caption_text
        return result
