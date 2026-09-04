from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field

class QueryIntent(str, Enum):
    VQA = "VQA"
    CAPTIONING = "CAPTIONING"
    GROUNDING = "GROUNDING"
    BI_TEMPORAL_CHANGE = "BI_TEMPORAL_CHANGE"
    OPTICAL_SAR = "OPTICAL_SAR"

class IntentClassificationResult(BaseModel):
    intent: QueryIntent
    confidence: float = Field(..., ge=0.0, le=1.0, description="Classification confidence score between 0.0 and 1.0")
    explanation: str = Field("", description="Rationale for intent assignment")


class LLMClassifierInterface(ABC):
    """
    Extensible LLM-backed intent classifier contract.
    Production deployments can plug in an LLM adapter that implements this interface.
    """

    @abstractmethod
    def classify(self, query: str, image_ids: Optional[List[str]] = None) -> IntentClassificationResult:
        raise NotImplementedError
