from typing import Any, Dict, List, Optional

from app.agent.adapters import geochat_adapter
from app.exceptions import MissingImageIdError


class GeoChatModelAdapter:
    """Thin wrapper around the remote GeoChat HTTP adapter."""

    def __init__(self):
        self.model_name = "geochat"

    def answer_question(self, image_ids: List[str], question: str) -> Dict[str, Any]:
        image_id = _first_image_id(image_ids, "GeoChat VQA")
        return geochat_adapter.run_geochat_vqa(image_id, question)

    def generate_caption(
        self,
        image_ids: List[str],
        prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        image_id = _first_image_id(image_ids, "GeoChat captioning")
        return geochat_adapter.run_geochat_caption(image_id, prompt)

    def ground_target(self, image_ids: List[str], query: str) -> Dict[str, Any]:
        image_id = _first_image_id(image_ids, "GeoChat grounding")
        return geochat_adapter.run_geochat_grounding(image_id, query)


def _first_image_id(image_ids: List[str], action: str) -> str:
    if not image_ids or not str(image_ids[0]).strip():
        raise MissingImageIdError(f"{action} requires an image ID.")
    return str(image_ids[0]).strip()


geochat_model = GeoChatModelAdapter()
