from typing import Any, Dict, List

from app.sih_raster import fuse_model_outputs
from app.tools.base import BaseTool


class EvidenceFusionTool(BaseTool):
    @property
    def name(self) -> str:
        return "EvidenceFusion"

    @property
    def description(self) -> str:
        return "Fuses multi-model outputs using the existing Sih fusion/evidence layer."

    @property
    def required_inputs(self) -> List[str]:
        return []

    def execute(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        tool_outputs = payload.get("tool_outputs", {})
        metadata = payload.get("metadata", {})
        spatial_analysis = payload.get("spatial_analysis", {})

        fused = fuse_model_outputs(tool_outputs)
        fused["temporal_evidence"] = {
            "image_count": metadata.get("metadata_count"),
            "images": metadata.get("images_metadata"),
        }
        fused["spatial_evidence"] = spatial_analysis
        fused["metadata_relevance"] = "Sih fusion/evidence.fuse_evidence"
        return fused
