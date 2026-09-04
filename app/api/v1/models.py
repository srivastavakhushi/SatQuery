from fastapi import APIRouter, HTTPException

from app.agent.adapters import popeye_adapter, remote
from app.exceptions import QueryPipelineError
from app.schemas.models import (
    CDChatChangeRequest,
    GeoChatCaptionRequest,
    GeoChatGroundingRequest,
    GeoChatVQARequest,
    PopeyeOpticalSarRequest,
    ResNetFeaturesRequest,
)
from app.tools.models.cd_chat import cd_chat_model
from app.tools.models.geochat import geochat_model
from app.tools.models.popeye import popeye_model
from app.tools.models.resnet import resnet_model

router = APIRouter()


@router.get("/models/health")
def models_health():
    return {
        "geochat": remote.probe_health("geochat"),
        "cdchat": remote.probe_health("cdchat"),
        "popeye": remote.probe_health("popeye"),
        "resnet": remote.probe_health("resnet"),
    }


@router.post("/models/geochat/vqa")
def geochat_vqa(request: GeoChatVQARequest):
    return _call(lambda: geochat_model.answer_question([request.image_id], request.question))


@router.post("/models/geochat/caption")
def geochat_caption(request: GeoChatCaptionRequest):
    return _call(lambda: geochat_model.generate_caption([request.image_id], prompt=request.prompt))


@router.post("/models/geochat/grounding")
def geochat_grounding(request: GeoChatGroundingRequest):
    return _call(lambda: geochat_model.ground_target([request.image_id], request.query))


@router.post("/models/cdchat/change")
def cdchat_change(request: CDChatChangeRequest):
    return _call(
        lambda: cd_chat_model.detect_changes(
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
