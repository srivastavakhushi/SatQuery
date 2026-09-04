from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, status
from app.schemas import ReportRequest, ReportResponse
from app.logger import execution_trace_logger
from app.tools import tool_registry

router = APIRouter()

SUPPORTED_FORMATS = {"json", "markdown", "md", "pdf"}

@router.post("/report", response_model=ReportResponse, status_code=status.HTTP_200_OK)
async def generate_report(request: ReportRequest):
    """
    POST /api/v1/report
    Generates a structured report from a previous execution trace ID or image session.
    """
    report_format = (request.format or "json").lower().strip()
    if report_format not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported report format '{request.format}'. Use json, markdown, or pdf."
        )

    trace_data = None
    if request.trace_id:
        trace_data = execution_trace_logger.get_trace(request.trace_id)
        if not trace_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Execution trace ID '{request.trace_id}' not found."
            )

    image_ids = request.image_ids or []
    consolidated = []
    if trace_data:
        consolidated.extend([
            f"Audit Trace ID: {trace_data.trace_id}",
            f"Query: {trace_data.query}",
            f"Latency: {trace_data.latency}s",
            f"Confidence Score: {trace_data.confidence}",
        ])
    if image_ids:
        consolidated.append(f"Session image IDs: {', '.join(image_ids)}")

    if trace_data:
        final_answer = f"Report generated for query: '{trace_data.query}'"
        intent = trace_data.selected_task
        models_used = trace_data.models_dispatched
    else:
        final_answer = "Multi-modal analysis report."
        intent = "SESSION_REPORT"
        models_used = ["GeoChat", "CD Chat", "RingMoGPT"]

    report_payload = {
        "intent": intent,
        "final_answer": final_answer,
        "fused_evidence": {"consolidated_evidence": consolidated},
        "models_used": models_used,
        "title": request.title,
        "format": report_format,
        "image_ids": image_ids,
        "trace_id": request.trace_id,
    }

    report_result = tool_registry.execute_tool("ReportGenerator", report_payload)

    return ReportResponse(
        report_id=report_result["report_id"],
        title=request.title or report_result["title"],
        generated_at=datetime.now(timezone.utc),
        intent=report_result.get("intent"),
        summary=report_result.get("summary", ""),
        key_findings=report_result.get("key_findings", []),
        evidence_details=report_result.get("evidence_details", {}),
        models_used=report_result.get("models_used", []),
        trace_id=request.trace_id
    )
