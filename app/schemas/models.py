from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class GeoChatVQARequest(BaseModel):
    image_id: str = Field(..., description="Stored upload image ID")
    question: str = Field(..., description="Visual question for GeoChat")


class GeoChatCaptionRequest(BaseModel):
    image_id: str
    prompt: Optional[str] = Field(None, description="Optional caption prompt")


class GeoChatGroundingRequest(BaseModel):
    image_id: str
    query: str = Field(..., description="Object or region to localize")


class CDChatChangeRequest(BaseModel):
    image_id_1: str
    image_id_2: str
    question: str


class PopeyeOpticalSarRequest(BaseModel):
    question: str
    optical_image_id: Optional[str] = None
    sar_image_id: Optional[str] = None
    image_ids: Optional[List[str]] = None


class ResNetFeaturesRequest(BaseModel):
    image_id: str


class ModelHealthStatus(BaseModel):
    configured: bool
    reachable: bool
    mode: str


class ModelsHealthResponse(BaseModel):
    geochat: Dict[str, Any]
    cdchat: Dict[str, Any]
    popeye: Dict[str, Any]
    resnet: Dict[str, Any]
