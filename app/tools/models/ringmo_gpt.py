from typing import Dict, Any, List

class RingMoGPTModelAdapter:
    """
    Unused in the live SIH stack.

    RingMoGPT is retained in the repository for reference only. It is not
    registered on query routes and must not be used for captioning,
    grounding, or optical-SAR inference.
    """

    def __init__(self):
        self.model_name = "RingMoGPT-RS"

    def process_optical_sar(self, image_ids: List[str], query: str) -> Dict[str, Any]:
        return {
            "model": self.model_name,
            "status": "unused",
            "mock": True,
            "summary": "RingMoGPT is not part of the live model stack.",
        }

    def generate_caption(self, image_ids: List[str]) -> Dict[str, Any]:
        return {
            "model": self.model_name,
            "status": "unused",
            "mock": True,
            "caption": "RingMoGPT is not part of the live model stack.",
        }

ringmo_gpt_model = RingMoGPTModelAdapter()
