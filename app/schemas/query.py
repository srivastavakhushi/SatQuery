from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

class QueryRequest(BaseModel):
    query: str = Field(..., json_schema_extra={"example": "What changed between these two images?"}, description="Natural language user query")
    image_ids: List[str] = Field(default_factory=list, description="List of uploaded image IDs referenced in the query")
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Optional execution parameters")

class ExecutionTrace(BaseModel):
    trace_id: str = Field(..., description="Unique ID for audit execution trace")
    query: str = Field(..., description="Original user prompt")
    selected_task: str = Field(..., description="Identified task / intent")
    models_dispatched: List[str] = Field(..., description="Models / Tools executed in workflow")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Parameters passed during execution")
    latency: float = Field(..., description="Total latency in seconds")
    confidence: float = Field(..., description="Overall confidence score")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Trace creation timestamp")
    intent_confidence: Optional[float] = Field(None, description="Classifier confidence")
    selected_model: Optional[str] = Field(None, description="Primary model selected for the intent")
    image_ids: List[str] = Field(default_factory=list, description="Image IDs used for this query")
    rsicrc_execution_time: Optional[float] = Field(None, description="RSICRC inference time in seconds")
    rsicrc_result: Optional[Dict[str, Any]] = Field(None, description="Raw RSICRC / ChangeDetection output")
    fusion_result: Optional[Dict[str, Any]] = Field(None, description="Evidence fusion output")
    errors: List[str] = Field(default_factory=list, description="Non-fatal pipeline errors")
    execution_logs: List[str] = Field(default_factory=list, description="Node-level execution logs")

class QueryResponse(BaseModel):
    status: str = "success"
    query: str
    intent: str
    selected_model: Optional[str] = Field(None, description="Primary inference model used for this query")
    confidence: float
    answer: str
    fused_evidence: Dict[str, Any] = Field(default_factory=dict)
    models_dispatched: List[str]
    execution_trace: ExecutionTrace
