"""
CDChat inference wrapper.

Calls the existing CDChat entry points:
- cdchat.model.builder.load_pretrained_model
- cdchat.mm_utils.tokenizer_image_token / get_model_name_from_path
- cdchat.conversation.conv_templates
- same RGB 448px + BGR-swap preprocessing as cdchat.eval.batch_cdchat_vqa.eval_model

Does not modify CDChat model internals.
"""

from __future__ import annotations

import logging
import os
import threading
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional, Union

from PIL import Image

logger = logging.getLogger(__name__)

ImageInput = Union[str, Path, bytes, Image.Image]

_MODEL_LOCK = threading.Lock()
_MODEL_BUNDLE: Optional[Dict[str, Any]] = None


def _cdchat_repo_root() -> Path:
    # services/cdchat/inference.py -> Backend/cdchat
    return Path(__file__).resolve().parents[2] / "cdchat"


def _ensure_cdchat_on_path() -> None:
    import sys

    repo = str(_cdchat_repo_root())
    if repo not in sys.path:
        sys.path.insert(0, repo)


def _resolve_image(source: ImageInput) -> Image.Image:
    if isinstance(source, Image.Image):
        image = source
    elif isinstance(source, bytes):
        image = Image.open(BytesIO(source))
    else:
        image = Image.open(source)
    return image.convert("RGB")


def load_cdchat_model() -> Dict[str, Any]:
    """
    Lazy-load tokenizer/model/image_processor via load_pretrained_model.
    """
    global _MODEL_BUNDLE
    if _MODEL_BUNDLE is not None:
        return _MODEL_BUNDLE

    with _MODEL_LOCK:
        if _MODEL_BUNDLE is not None:
            return _MODEL_BUNDLE

        _ensure_cdchat_on_path()
        import torch
        from cdchat.model.builder import load_pretrained_model
        from cdchat.mm_utils import get_model_name_from_path
        from cdchat.utils import disable_torch_init

        model_path = os.environ.get("CDCHAT_MODEL_PATH", "").strip()
        model_base = os.environ.get("CDCHAT_MODEL_BASE", "").strip() or None
        mm_projector_path = os.environ.get("CDCHAT_MM_PROJECTOR_PATH", "").strip() or None
        device = os.environ.get("CDCHAT_DEVICE", "cuda").strip() or "cuda"

        if not model_path:
            raise RuntimeError("CDCHAT_MODEL_PATH is not configured.")
        if not Path(model_path).exists():
            raise RuntimeError(f"CDCHAT_MODEL_PATH does not exist: {model_path}")

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Set CDCHAT_DEVICE=cpu or attach a GPU.")

        disable_torch_init()
        model_name = get_model_name_from_path(os.path.expanduser(model_path))
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            os.path.expanduser(model_path),
            model_base,
            model_name,
            mm_projector_path=mm_projector_path,
        )

        _MODEL_BUNDLE = {
            "tokenizer": tokenizer,
            "model": model,
            "image_processor": image_processor,
            "context_len": context_len,
            "device": device,
        }
        logger.info("Loaded CDChat weights from %s", model_path)
        return _MODEL_BUNDLE


def predict_change(
    image1: ImageInput,
    image2: ImageInput,
    question: str,
    conv_mode: str = "llava_v1",
) -> str:
    """
    Single-pair inference mirroring eval_model() in batch_cdchat_vqa.py.
    """
    _ensure_cdchat_on_path()
    import torch
    from cdchat.constants import (
        DEFAULT_IM_END_TOKEN,
        DEFAULT_IM_START_TOKEN,
        DEFAULT_IMAGE_TOKEN,
        IMAGE_TOKEN_INDEX,
    )
    from cdchat.conversation import SeparatorStyle, conv_templates
    from cdchat.mm_utils import tokenizer_image_token

    bundle = load_cdchat_model()
    tokenizer = bundle["tokenizer"]
    model = bundle["model"]
    image_processor = bundle["image_processor"]
    device = bundle["device"]

    image_a = _resolve_image(image1)
    image_b = _resolve_image(image2)

    qs = question.strip()
    if getattr(model.config, "mm_use_im_start_end", False):
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(device)

    image_tensor_a = image_processor.preprocess(
        image_a,
        crop_size={"height": 448, "width": 448},
        size={"shortest_edge": 448},
        return_tensors="pt",
    )["pixel_values"][0]
    image_tensor_b = image_processor.preprocess(
        image_b,
        crop_size={"height": 448, "width": 448},
        size={"shortest_edge": 448},
        return_tensors="pt",
    )["pixel_values"][0]
    image_tensor_a = image_tensor_a[[2, 1, 0], :, :]
    image_tensor_b = image_tensor_b[[2, 1, 0], :, :]

    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    dummy_label = torch.zeros(1, image_a.size[1], image_a.size[0], dtype=torch.float32)
    image_tensor_batch = {
        "pre": torch.stack([image_tensor_a]).to(device=device, dtype=dtype),
        "post": torch.stack([image_tensor_b]).to(device=device, dtype=dtype),
        "targets": dummy_label.unsqueeze(0).to(device),
    }

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor_batch,
            do_sample=False,
            temperature=float(os.environ.get("CDCHAT_TEMPERATURE", "0.2")),
            top_p=None,
            num_beams=1,
            max_new_tokens=256,
            length_penalty=2.0,
            use_cache=True,
        )

    input_token_len = input_ids.shape[1]
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)
    output = outputs[0].strip()
    if stop_str and output.endswith(stop_str):
        output = output[: -len(stop_str)]
    return output.strip()
