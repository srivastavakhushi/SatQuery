from typing import Any, Dict, List

from app.agent.adapters import resnet_adapter
from app.exceptions import MissingImageIdError


class ResNetModelAdapter:
    """
    Supporting ResNet-50 feature/domain adapter.

    Not used by ChangeDetection or conversational routing.
    """

    def __init__(self):
        self.model_name = "resnet-50"

    def extract_features(self, image_ids: List[str]) -> Dict[str, Any]:
        if not image_ids or not str(image_ids[0]).strip():
            raise MissingImageIdError("ResNet feature extraction requires an image ID.")
        return resnet_adapter.run_resnet_features(str(image_ids[0]).strip())


resnet_model = ResNetModelAdapter()
