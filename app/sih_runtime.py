"""Put the nested Sih repository on sys.path so `raster.*` and `fusion.*` import."""

from __future__ import annotations

import sys
from pathlib import Path

from app.config import settings


def ensure_sih_on_path() -> Path:
    sih_root = Path(settings.SIH_DIR).resolve()
    if not sih_root.exists():
        raise FileNotFoundError(f"Sih folder not found at {sih_root}")
    sih_str = str(sih_root)
    if sih_str not in sys.path:
        sys.path.insert(0, sih_str)
    return sih_root
