"""
Resolve stored image_ids to RGB PNG bytes for remote model adapters.

Sih raster loading is used only when the stored file is not a simple RGB image
(for example a GeoTIFF). Model-specific CLIP resize stays on the remote GPU host.
"""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image, UnidentifiedImageError

from app.exceptions import InvalidImageFormatError, PreprocessingError
from app.storage import resolve_image_path

ImageInput = Union[str, Path, bytes, Image.Image, np.ndarray]


def image_id_to_base64_png(image_id: str) -> str:
    png = path_to_rgb_png_bytes(resolve_image_path(image_id))
    return base64.b64encode(png).decode("ascii")


def path_to_rgb_png_bytes(path: Path) -> bytes:
    try:
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            return _png_bytes(rgb)
    except (UnidentifiedImageError, OSError):
        pass
    except Exception as exc:
        raise PreprocessingError(f"Image conversion failed for '{path.name}'.") from exc

    try:
        from app.sih_raster import load_processed_stack

        processed = load_processed_stack(path)
        return _png_bytes(sih_array_to_rgb(processed))
    except (InvalidImageFormatError, PreprocessingError):
        raise
    except Exception as exc:
        raise InvalidImageFormatError(
            f"'{path.name}' could not be decoded as an image for model inference."
        ) from exc


def source_to_rgb_png_bytes(source: ImageInput, label: str) -> bytes:
    try:
        if isinstance(source, np.ndarray):
            image = sih_array_to_rgb(source)
        elif isinstance(source, Image.Image):
            image = source.convert("RGB") if source.mode != "RGB" else source
        elif isinstance(source, bytes):
            image = Image.open(BytesIO(source)).convert("RGB")
        else:
            path = Path(source)
            if not path.exists():
                raise PreprocessingError(f"{label} file was not found for model conversion.")
            return path_to_rgb_png_bytes(path)
        return _png_bytes(image)
    except (InvalidImageFormatError, PreprocessingError):
        raise
    except UnidentifiedImageError as exc:
        raise InvalidImageFormatError(
            f"{label} could not be decoded as an image."
        ) from exc
    except Exception as exc:
        raise PreprocessingError(f"{label} conversion failed.") from exc


def sih_array_to_rgb(processed: np.ndarray) -> Image.Image:
    """
    Map Sih preprocess_image output to RGB.

    13-band Sentinel-2: B04, B03, B02 (indices 3, 2, 1 in BAND_NAMES).
    3-band stacks: first three channels.
    """
    if processed.ndim != 3:
        raise PreprocessingError("Sih processed array must have shape (bands, height, width).")
    band_count = processed.shape[0]
    if band_count >= 13:
        rgb = np.stack([processed[3], processed[2], processed[1]], axis=-1)
    elif band_count >= 3:
        rgb = np.stack([processed[0], processed[1], processed[2]], axis=-1)
    else:
        gray = processed[0]
        rgb = np.stack([gray, gray, gray], axis=-1)
    uint8 = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    return Image.fromarray(uint8, mode="RGB")


def _png_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
