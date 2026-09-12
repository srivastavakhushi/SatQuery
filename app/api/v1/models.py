from fastapi import APIRouter, HTTPException

from app.agent.adapters import popeye_adapter, remote
from app.exceptions import QueryPipelineError
from app.schemas.models import (
    LLaVACaptionRequest,
    LLaVAGroundingRequest,
    LLaVAVQARequest,
    PopeyeOpticalSarRequest,
    ResNetFeaturesRequest,
    RSICRCChangeRequest,
)
from app.tools.models.llava import llava_model
from app.tools.models.popeye import popeye_model
from app.tools.models.resnet import resnet_model
from app.tools.models.rsicrc import rsicrc_model

router = APIRouter()


@router.get("/models/health")
def models_health():
    return {
        "llava": remote.probe_health("llava"),
        "rsicrc": remote.probe_health("rsicrc"),
        "popeye": remote.probe_health("popeye"),
        "resnet": remote.probe_health("resnet"),
    }


@router.post("/models/llava/vqa")
def llava_vqa(request: LLaVAVQARequest):
    return _call(lambda: llava_model.answer_question([request.image_id], request.question))


@router.post("/models/llava/caption")
def llava_caption(request: LLaVACaptionRequest):
    return _call(lambda: llava_model.generate_caption([request.image_id], prompt=request.prompt))


@router.post("/models/llava/grounding")
def llava_grounding(request: LLaVAGroundingRequest):
    return _call(lambda: llava_model.ground_target([request.image_id], request.query))


@router.post("/models/rsicrc/change")
def rsicrc_change(request: RSICRCChangeRequest):
    return _call(
        lambda: rsicrc_model.detect_changes(
            [request.image_id_1, request.image_id_2],
            request.question,
        )
    )


@router.post("/models/popeye/optical-sar")
def popeye_optical_sar(request: PopeyeOpticalSarRequest):
    if request.optical_image_id or request.sar_image_id:
        return _call(
            lambda: popeye_adapter.run_popeye(
                question=request.question,
                optical_image_id=request.optical_image_id,
                sar_image_id=request.sar_image_id,
            )
        )
    return _call(lambda: popeye_model.process_optical_sar(request.image_ids or [], request.question))


@router.post("/models/resnet/features")
def resnet_features(request: ResNetFeaturesRequest):
    return _call(lambda: resnet_model.extract_features([request.image_id]))


def _call(fn):
    try:
        return fn()
    except QueryPipelineError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from None
