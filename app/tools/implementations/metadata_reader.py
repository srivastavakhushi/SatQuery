from typing import Dict, Any, List

from app.sih_raster import raster_metadata
from app.storage import resolve_image_path
from app.tools.base import BaseTool


class MetadataReaderTool(BaseTool):
    @property
    def name(self) -> str:
        return "MetadataReader"

    @property
    def description(self) -> str:
        return "Reads raster metadata via Sih raster.metadata.get_raster_metadata."

    @property
    def required_inputs(self) -> List[str]:
        return []

    def execute(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        image_ids = payload.get("image_ids", [])
        images_metadata = []
        for image_id in image_ids:
            try:
                path = resolve_image_path(image_id)
                filepath = str(path)
            except Exception:
                path = None
                filepath = None
            raster_meta = raster_metadata(path) if path is not None else None
            images_metadata.append({
                "image_id": image_id,
                "filepath": filepath,
                "raster_metadata": raster_meta,
                "width": (raster_meta or {}).get("width"),
                "height": (raster_meta or {}).get("height"),
                "band_count": (raster_meta or {}).get("band_count"),
                "crs": (raster_meta or {}).get("crs"),
            })
        return {
            "status": "success",
            "metadata_count": len(images_metadata),
            "images_metadata": images_metadata,
        }
