from typing import Any, Dict, List

from app.agent.adapters import cdchat_adapter
from app.exceptions import InsufficientImagesError
from app.sih_raster import preprocess_temporal_pair
from app.storage import resolve_image_path


class CDChatModelAdapter:
    """
    Resolves stored image IDs, runs Sih raster preprocessing, then CDChat.
    Does not run on upload.
    """

    def __init__(self):
        self.model_name = "cdchat"

    def detect_changes(self, image_ids: List[str], query: str) -> Dict[str, Any]:
        if not image_ids or len(image_ids) < 2:
            raise InsufficientImagesError(
                "Bi-temporal change analysis requires two image IDs."
            )

        path1 = resolve_image_path(image_ids[0])
        path2 = resolve_image_path(image_ids[1])
        processed1, processed2 = preprocess_temporal_pair(path1, path2)

        result = cdchat_adapter.run_cdchat(
            image1=processed1,
            image2=processed2,
            question=query,
        )

        answer = result.get("answer") or result.get("summary") or ""
        merged = dict(result)
        merged.update(
            {
                "model": result.get("model", self.model_name),
                "status": "success",
                "confidence": result.get("confidence", 0.9),
                "answer": answer,
                "summary": result.get("summary") or answer,
                "image_ids": [image_ids[0], image_ids[1]],
                "task": "bi_temporal_change_detection",
            }
        )
        return merged


cd_chat_model = CDChatModelAdapter()
