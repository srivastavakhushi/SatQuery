from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse

from app.logger import execution_trace_logger
from app.schemas import ReportRequest, ReportResponse
from app.tools import tool_registry

router = APIRouter()

SUPPORTED_FORMATS = {"json", "markdown", "md", "pdf"}

_INFERENCE_MODELS = ("RSICRC", "GeoLLaVA", "Popeye", "ResNet-50", "ResNet")


@router.post("/report")
async def generate_report(request: ReportRequest):
    """
    POST /api/v1/report
    Generates a structured report from a previous execution trace ID or image session.
    PDF is the default download format.
    """
    report_format = (request.format or "pdf").lower().strip()
    if report_format not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported report format '{request.format}'. Use pdf, markdown, or json."
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
    if trace_data:
        image_ids = image_ids or list(trace_data.image_ids or [])

    selected_model = _primary_model(trace_data)
    summary = _trace_summary(trace_data)
    models_used = list(trace_data.models_dispatched) if trace_data else []
    intent = trace_data.selected_task if trace_data else "SESSION_REPORT"
    query = trace_data.query if trace_data else ""

    consolidated = []
    if selected_model:
        consolidated.append(f"Primary model: {selected_model}")
    if query:
        consolidated.append(f"Query: {query}")
    if trace_data:
        consolidated.extend([
            f"Audit Trace ID: {trace_data.trace_id}",
            f"Latency: {trace_data.latency}s",
            f"Confidence Score: {trace_data.confidence}",
        ])
        fusion = trace_data.fusion_result or {}
        for item in fusion.get("consolidated_evidence") or []:
            consolidated.append(str(item))
    if image_ids:
        consolidated.append(f"Session image IDs: {', '.join(image_ids)}")

    report_payload = {
        "intent": intent,
        "final_answer": summary,
        "fused_evidence": {"consolidated_evidence": consolidated},
        "models_used": models_used,
        "title": request.title,
        "format": report_format,
        "image_ids": image_ids,
        "trace_id": request.trace_id,
        "selected_model": selected_model,
        "query": query,
    }

    report_result = tool_registry.execute_tool("ReportGenerator", report_payload)
    file_path = Path(report_result["file_path"])

    if report_format == "pdf":
        if not file_path.exists():
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="PDF report was generated but the file could not be found.",
            )
        return FileResponse(
            path=str(file_path),
            media_type="application/pdf",
            filename=f"{report_result['report_id']}.pdf",
        )

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


def _primary_model(trace_data) -> str:
    if not trace_data:
        return ""
    if trace_data.selected_model:
        return str(trace_data.selected_model)
    for name in trace_data.models_dispatched or []:
        if name in _INFERENCE_MODELS:
            return name
    return ""


def _trace_summary(trace_data) -> str:
    if not trace_data:
        return "Multi-modal analysis report."
    fusion = trace_data.fusion_result or {}
    model_block = fusion.get("rsicrc") or {}
    answer = model_block.get("answer") or fusion.get("decision")
    if not answer:
        result = trace_data.rsicrc_result or {}
        answer = result.get("answer") or result.get("summary")
    if answer:
        return str(answer)
    return f"Analysis completed for query: '{trace_data.query}'"
