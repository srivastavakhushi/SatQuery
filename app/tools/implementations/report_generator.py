from typing import Dict, Any, List
import json
import uuid
from datetime import datetime, timezone
from app.config import settings
from app.tools.base import BaseTool


def _build_markdown(report: Dict[str, Any]) -> str:
    findings = report.get("key_findings") or []
    findings_md = "\n".join(f"- {item}" for item in findings) or "- None"
    models = ", ".join(report.get("models_used") or []) or "N/A"
    return (
        f"# {report.get('title')}\n\n"
        f"- Report ID: `{report.get('report_id')}`\n"
        f"- Intent: `{report.get('intent')}`\n"
        f"- Generated: {report.get('generated_at')}\n"
        f"- Models: {models}\n\n"
        f"## Summary\n\n{report.get('summary') or 'N/A'}\n\n"
        f"## Key Findings\n\n{findings_md}\n"
    )


def _build_simple_pdf(text: str) -> bytes:
    """Write a minimal single-page PDF without third-party dependencies."""
    escaped = (
        text.replace("\\", "\\\\")
        .replace("(", "\\(")
        .replace(")", "\\)")
        .replace("\r", "")
    )
    lines = escaped.split("\n")[:60]
    content_lines = ["BT", "/F1 11 Tf", "50 780 Td", "14 TL"]
    for line in lines:
        content_lines.append(f"({line[:110]}) '")
    content_lines.append("ET")
    stream = "\n".join(content_lines).encode("latin-1", errors="replace")

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]

    buffer = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(buffer))
        buffer.extend(f"{index} 0 obj\n".encode())
        buffer.extend(obj)
        buffer.extend(b"\nendobj\n")

    xref_pos = len(buffer)
    buffer.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    buffer.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        buffer.extend(f"{offset:010d} 00000 n \n".encode())
    buffer.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n".encode()
    )
    return bytes(buffer)


class ReportGeneratorTool(BaseTool):
    @property
    def name(self) -> str:
        return "ReportGenerator"

    @property
    def description(self) -> str:
        return "Generates structured analysis reports summarizing intelligence findings and audit traces."

    @property
    def required_inputs(self) -> List[str]:
        return []

    def execute(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        intent = payload.get("intent", "GENERAL_ANALYSIS")
        final_answer = payload.get("final_answer", "")
        fused_evidence = payload.get("fused_evidence", {})
        models_used = payload.get("models_used", [])
        requested_format = (payload.get("format") or "json").lower().strip()
        custom_title = payload.get("title")

        report_id = f"REP-{uuid.uuid4().hex[:8].upper()}"
        report = {
            "report_id": report_id,
            "title": custom_title or f"Multi-Modal Intelligence Report - {intent}",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "intent": intent,
            "summary": final_answer,
            "key_findings": fused_evidence.get("consolidated_evidence", []),
            "evidence_details": fused_evidence,
            "models_used": models_used,
            "image_ids": payload.get("image_ids") or [],
            "trace_id": payload.get("trace_id"),
        }

        settings.REPORT_DIR.mkdir(parents=True, exist_ok=True)
        if requested_format in {"markdown", "md"}:
            path = settings.REPORT_DIR / f"{report_id}.md"
            path.write_text(_build_markdown(report), encoding="utf-8")
        elif requested_format == "pdf":
            path = settings.REPORT_DIR / f"{report_id}.pdf"
            path.write_bytes(_build_simple_pdf(_build_markdown(report)))
        else:
            path = settings.REPORT_DIR / f"{report_id}.json"
            path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        report["file_path"] = str(path.resolve())
        report["format"] = "markdown" if requested_format in {"markdown", "md"} else requested_format
        return report
