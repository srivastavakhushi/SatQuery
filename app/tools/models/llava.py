from typing import Any, Dict, List, Optional

from app.agent.adapters import llava_adapter
from app.exceptions import MissingImageIdError


class LLaVAModelAdapter:
    """Thin wrapper around the remote GeoLLaVA HTTP adapter."""

    def __init__(self):
        self.model_name = "llava"

    def answer_question(self, image_ids: List[str], question: str) -> Dict[str, Any]:
        image_id = _first_image_id(image_ids, "GeoLLaVA VQA")
        return llava_adapter.run_llava_vqa(image_id, question)

    def generate_caption(
        self,
        image_ids: List[str],
        prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        image_id = _first_image_id(image_ids, "GeoLLaVA captioning")
        return llava_adapter.run_llava_caption(image_id, prompt)

    def ground_target(self, image_ids: List[str], query: str) -> Dict[str, Any]:
        image_id = _first_image_id(image_ids, "GeoLLaVA grounding")
        return llava_adapter.run_llava_grounding(image_id, query)


def _first_image_id(image_ids: List[str], action: str) -> str:
    if not image_ids or not str(image_ids[0]).strip():
        raise MissingImageIdError(f"{action} requires an image ID.")
    return str(image_ids[0]).strip()


llava_model = LLaVAModelAdapter()
