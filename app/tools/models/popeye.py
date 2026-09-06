from typing import Any, Dict, List

from app.agent.adapters import popeye_adapter
from app.exceptions import MissingImageIdError
from app.image_modalities import split_optical_sar_ids


class PopeyeModelAdapter:
    """Thin wrapper around the remote Popeye optical+SAR HTTP adapter."""

    def __init__(self):
        self.model_name = "popeye"

    def process_optical_sar(self, image_ids: List[str], query: str) -> Dict[str, Any]:
        optical_id, sar_id = split_optical_sar_ids(image_ids)
        if not optical_id and not sar_id:
            raise MissingImageIdError(
                "Popeye optical-SAR analysis requires at least one image ID."
            )
        return popeye_adapter.run_popeye(
            question=query,
            optical_image_id=optical_id,
            sar_image_id=sar_id,
        )


popeye_model = PopeyeModelAdapter()
