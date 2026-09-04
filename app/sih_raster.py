"""
Thin callers around existing Sih raster / fusion functions.

Does not reimplement GeoTIFF reading, band normalization, or fusion math.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.exceptions import InvalidImageFormatError, PreprocessingError
from app.sih_runtime import ensure_sih_on_path

logger = logging.getLogger(__name__)


def _load_raw_stack(source: Path) -> np.ndarray:
    """
    Load a stored upload into a (bands, height, width) array.

    OSCD-style folders use Sih `load_band_stack` when Rasterio is available.
    Single files use Rasterio (same library as Sih `get_band`) or Pillow.
    """
    path = Path(source)
    if path.is_dir():
        ensure_sih_on_path()
        from raster.bands import load_band_stack

        try:
            return load_band_stack(str(path))
        except Exception as exc:
            raise PreprocessingError("Sih could not load the band stack.") from exc

    try:
        import rasterio

        with rasterio.open(path) as src:
            array = src.read()
        if array.ndim == 2:
            array = array[np.newaxis, ...]
        return array
    except Exception:
        pass

    try:
        from PIL import Image

        with Image.open(path) as image:
            array = np.array(image)
    except Exception as exc:
        raise InvalidImageFormatError("Image could not be read for Sih preprocessing.") from exc

    if array.ndim == 2:
        array = array[np.newaxis, ...]
    elif array.ndim == 3:
        array = np.transpose(array, (2, 0, 1))
    else:
        raise InvalidImageFormatError("Unsupported image array rank for Sih preprocessing.")
    return array


def _sih_preprocess_image(image: np.ndarray) -> np.ndarray:
    """Sih `raster.model_input.preprocess_image`: per-band `normalize_band`."""
    ensure_sih_on_path()
    from raster.preprocessing import normalize_band

    if image.ndim == 2:
        image = image[np.newaxis, ...]
    processed_bands = [normalize_band(band) for band in image]
    return np.stack(processed_bands, axis=0)


def load_processed_stack(source: Path) -> np.ndarray:
    """Load one stored file and apply Sih per-band normalize_band."""
    try:
        return _sih_preprocess_image(_load_raw_stack(Path(source)))
    except (InvalidImageFormatError, PreprocessingError):
        raise
    except Exception as exc:
        raise PreprocessingError("Sih raster preprocessing failed.") from exc


def preprocess_temporal_pair(path1: Path, path2: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Query-time Sih preprocessing for two stored images.

    Calls raster.preprocessing.validate_temporal_pair and normalize_band
    (the same path as raster.model_input.preprocess_image).
    """
    ensure_sih_on_path()
    from raster.preprocessing import validate_temporal_pair

    try:
        raw1 = _load_raw_stack(Path(path1))
        raw2 = _load_raw_stack(Path(path2))
        validate_temporal_pair(raw1, raw2)
        return _sih_preprocess_image(raw1), _sih_preprocess_image(raw2)
    except (InvalidImageFormatError, PreprocessingError):
        raise
    except Exception as exc:
        raise PreprocessingError("Sih raster preprocessing failed.") from exc


def raster_metadata(file_path: Path) -> Optional[Dict[str, Any]]:
    try:
        ensure_sih_on_path()
        from raster.metadata import get_raster_metadata

        return get_raster_metadata(str(file_path))
    except Exception as exc:
        logger.debug("Sih get_raster_metadata skipped for %s: %s", file_path, exc)
        return None


def pair_alignment(path1: Path, path2: Path) -> Dict[str, Any]:
    """Sih validate_temporal_pair + check_alignment on preprocessed first bands."""
    result: Dict[str, Any] = {
        "analysis_type": "Bi-Temporal Overlap Alignment",
        "engine": "Sih/raster.alignment.check_alignment",
    }
    try:
        ensure_sih_on_path()
        from raster.alignment import check_alignment

        processed1, processed2 = preprocess_temporal_pair(path1, path2)
        mean_difference, max_difference = check_alignment(processed1[0], processed2[0])
        result.update(
            {
                "status": "completed",
                "shape": list(processed1.shape),
                "mean_pixel_difference": float(mean_difference),
                "max_pixel_difference": float(max_difference),
            }
        )
        meta = raster_metadata(Path(path1)) or {}
        if meta.get("crs"):
            result["coordinate_system"] = str(meta["crs"])
        return result
    except Exception as exc:
        logger.info("Sih alignment skipped: %s", exc)
        result["status"] = "skipped"
        result["detail"] = "Temporal pair alignment could not be computed."
        return result


def fuse_model_outputs(tool_outputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Wrap tool dicts as Sih ModelOutput, then call
    evidence_from_model_output + fuse_evidence.
    """
    ensure_sih_on_path()
    from fusion.evidence import evidence_from_model_output, fuse_evidence
    from raster.model_output import create_model_output

    evidence_items: List[Any] = []
    cdchat_payload = None

    for tool_name, output in (tool_outputs or {}).items():
        if not isinstance(output, dict):
            continue
        model_name = str(output.get("model") or tool_name)
        if tool_name == "ChangeDetection" or model_name.lower().startswith("cdchat"):
            cdchat_payload = output
        confidence = output.get("confidence")
        if confidence is None:
            continue
        prediction = output.get("prediction")
        if not isinstance(prediction, np.ndarray):
            prediction = np.zeros((1, 1), dtype=np.uint8)
        model_output = create_model_output(
            model_name=model_name,
            task=str(output.get("task") or tool_name),
            prediction=prediction,
            confidence=float(confidence),
            metadata={
                "answer": output.get("answer") or output.get("summary"),
                "image_ids": output.get("image_ids"),
            },
        )
        evidence_items.append(evidence_from_model_output(model_output))

    if not evidence_items:
        raise ValueError("No fusable evidence with confidence scores was produced.")

    fused = fuse_evidence(evidence_items)
    consolidated = [
        f"[{item.source}]: {item.description} (score={item.score})"
        for item in fused.evidence
    ]
    if cdchat_payload:
        answer = cdchat_payload.get("answer") or cdchat_payload.get("summary")
        if answer:
            consolidated.insert(0, f"[ChangeDetection]: {answer}")

    return {
        "status": "success",
        "decision": fused.decision,
        "fusion_confidence": float(fused.final_score),
        "fused_evidence_count": len(fused.evidence),
        "consolidated_evidence": consolidated,
        "sih_evidence": [
            {"source": item.source, "score": item.score, "description": item.description}
            for item in fused.evidence
        ],
        "cdchat": (
            {
                "answer": cdchat_payload.get("answer") or cdchat_payload.get("summary"),
                "confidence": cdchat_payload.get("confidence"),
                "model": cdchat_payload.get("model"),
                "elapsed_seconds": cdchat_payload.get("elapsed_seconds"),
                "image_ids": cdchat_payload.get("image_ids"),
            }
            if cdchat_payload
            else None
        ),
    }
