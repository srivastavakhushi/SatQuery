from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

class ReportRequest(BaseModel):
    trace_id: Optional[str] = Field(None, description="Execution trace ID to generate report for")
    image_ids: Optional[List[str]] = Field(default_factory=list, description="List of image IDs for session report")
    title: Optional[str] = Field("Geospatial & Multi-Modal Intelligence Report", description="Custom report title")
    format: str = Field("json", description="Report format: json, markdown, or pdf")

class ReportResponse(BaseModel):
    report_id: str
    title: str
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    intent: Optional[str] = None
    summary: str
    key_findings: List[str]
    evidence_details: Dict[str, Any]
    models_used: List[str]
    trace_id: Optional[str] = None
