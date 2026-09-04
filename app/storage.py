from pathlib import Path
from typing import List, Optional

from app.config import settings
from app.exceptions import ImageNotFoundError, MissingImageIdError


def resolve_image_path(image_id: str) -> Path:
    """Map an upload file_id to the stored original file."""
    if image_id is None or not str(image_id).strip():
        raise MissingImageIdError("Image ID is missing.")

    safe_id = Path(str(image_id).strip()).name
    if not safe_id or safe_id != str(image_id).strip():
        raise ImageNotFoundError(f"Image '{image_id}' was not found.")

    upload_dir = Path(settings.UPLOAD_DIR)
    if not upload_dir.exists():
        raise ImageNotFoundError(f"Image '{safe_id}' was not found.")

    matches: List[Path] = sorted(upload_dir.glob(f"{safe_id}_*"))
    if not matches:
        raise ImageNotFoundError(f"Image '{safe_id}' was not found.")
    return matches[0]


def resolve_image_paths(image_ids: Optional[List[str]]) -> List[Path]:
    return [resolve_image_path(image_id) for image_id in (image_ids or [])]
