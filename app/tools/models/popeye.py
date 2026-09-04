from typing import Any, Dict, List, Optional

from app.agent.adapters import popeye_adapter
from app.exceptions import MissingImageIdError


class PopeyeModelAdapter:
    """Thin wrapper around the remote Popeye optical+SAR HTTP adapter."""

    def __init__(self):
        self.model_name = "popeye"

    def process_optical_sar(self, image_ids: List[str], query: str) -> Dict[str, Any]:
        optical_id, sar_id = _split_optical_sar_ids(image_ids)
        return popeye_adapter.run_popeye(
            question=query,
            optical_image_id=optical_id,
            sar_image_id=sar_id,
        )


def _split_optical_sar_ids(image_ids: Optional[List[str]]) -> tuple:
    ids = [str(item).strip() for item in (image_ids or []) if str(item).strip()]
    if not ids:
        raise MissingImageIdError(
            "Popeye optical-SAR analysis requires at least one image ID."
        )
    if len(ids) == 1:
        if "sar" in ids[0].lower():
            return None, ids[0]
        return ids[0], None
    return ids[0], ids[1]


popeye_model = PopeyeModelAdapter()
