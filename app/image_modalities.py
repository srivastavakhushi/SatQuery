"""Detect optical vs SAR uploads from filenames and band counts."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.storage import resolve_image_path

_SAR_TOKEN = re.compile(
    r"(?:^|[_\-\s.])(?:s1|sar|sentinel[\-_]?1)(?:[_\-\s.]|$)",
    re.IGNORECASE,
)
_OPTICAL_TOKEN = re.compile(
    r"(?:^|[_\-\s.])(?:s2|optical|sentinel[\-_]?2)(?:[_\-\s.]|$)",
    re.IGNORECASE,
)


def suggests_optical_sar(image_ids: Optional[Sequence[str]]) -> bool:
    hints = [_hint(image_id) for image_id in _clean_ids(image_ids)]
    if not hints:
        return False
    if any(item["is_sar"] for item in hints):
        return True
    if len(hints) < 2:
        return False
    bands = [item["band_count"] for item in hints[:2]]
    if None in bands:
        return False
    return min(bands) == 1 and max(bands) >= 3


def split_optical_sar_ids(
    image_ids: Optional[Sequence[str]],
) -> Tuple[Optional[str], Optional[str]]:
    ids = _clean_ids(image_ids)
    if not ids:
        return None, None

    hints = [_hint(image_id) for image_id in ids]
    sar_ids = [item["image_id"] for item in hints if item["is_sar"]]
    optical_ids = [
        item["image_id"]
        for item in hints
        if item["is_optical"] and not item["is_sar"]
    ]
    if not optical_ids:
        optical_ids = [
            item["image_id"]
            for item in hints
            if item["band_count"] is not None
            and item["band_count"] >= 3
            and not item["is_sar"]
        ]
    if not sar_ids:
        sar_ids = [
            item["image_id"]
            for item in hints
            if item["band_count"] == 1 and not item["is_optical"]
        ]

    optical_id = optical_ids[0] if optical_ids else None
    sar_id = sar_ids[0] if sar_ids else None
    leftover = [image_id for image_id in ids if image_id not in {optical_id, sar_id}]
    if optical_id is None and leftover:
        optical_id = leftover.pop(0)
    if sar_id is None and leftover:
        sar_id = leftover.pop(0)
    if optical_id is None and sar_id is None:
        return (ids[0], ids[1] if len(ids) > 1 else None)
    return optical_id, sar_id


def _clean_ids(image_ids: Optional[Sequence[str]]) -> List[str]:
    return [str(item).strip() for item in (image_ids or []) if str(item).strip()]


def _hint(image_id: str) -> Dict[str, Any]:
    filename = image_id
    band_count: Optional[int] = None
    try:
        path = resolve_image_path(image_id)
        filename = path.name
        band_count = _band_count(path)
    except Exception:
        pass
    text = f"{image_id} {filename}".lower()
    return {
        "image_id": image_id,
        "is_sar": bool(_SAR_TOKEN.search(text)),
        "is_optical": bool(_OPTICAL_TOKEN.search(text)),
        "band_count": band_count,
    }


def _band_count(path: Path) -> Optional[int]:
    try:
        from PIL import Image

        with Image.open(path) as image:
            return len(image.getbands())
    except Exception:
        return None
