import json
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from pathlib import Path
from app.config import settings
from app.schemas.query import ExecutionTrace

class ExecutionTraceLogger:
    """
    Execution Trace Logger for Auditability.
    Records execution parameters, model dispatches, latency, and confidence scores.
    """

    def __init__(self, trace_dir: Optional[Path] = None):
        self.trace_dir = trace_dir or settings.TRACE_DIR
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self._in_memory_traces: Dict[str, ExecutionTrace] = {}

    def log_trace(
        self,
        query: str,
        selected_task: str,
        models_dispatched: List[str],
        parameters: Dict[str, Any],
        latency: float,
        confidence: float,
        trace_id: Optional[str] = None,
        intent_confidence: Optional[float] = None,
        selected_model: Optional[str] = None,
        image_ids: Optional[List[str]] = None,
        cdchat_execution_time: Optional[float] = None,
        cdchat_result: Optional[Dict[str, Any]] = None,
        fusion_result: Optional[Dict[str, Any]] = None,
        errors: Optional[List[str]] = None,
        execution_logs: Optional[List[str]] = None,
    ) -> ExecutionTrace:
        if not trace_id:
            trace_id = f"trc-{uuid.uuid4().hex[:12]}"

        trace = ExecutionTrace(
            trace_id=trace_id,
            query=query,
            selected_task=selected_task,
            models_dispatched=models_dispatched,
            parameters=parameters or {},
            latency=round(latency, 3),
            confidence=round(confidence, 2),
            timestamp=datetime.now(timezone.utc),
            intent_confidence=intent_confidence,
            selected_model=selected_model,
            image_ids=image_ids or [],
            cdchat_execution_time=cdchat_execution_time,
            cdchat_result=cdchat_result,
            fusion_result=fusion_result,
            errors=errors or [],
            execution_logs=execution_logs or [],
        )

        # Store in memory
        self._in_memory_traces[trace_id] = trace

        # Persist audit file to disk
        file_path = self.trace_dir / f"{trace_id}.json"
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(trace.model_dump(mode="json"), f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to persist trace to file {file_path}: {e}")

        return trace

    def get_trace(self, trace_id: str) -> Optional[ExecutionTrace]:
        if trace_id in self._in_memory_traces:
            return self._in_memory_traces[trace_id]

        file_path = self.trace_dir / f"{trace_id}.json"
        if file_path.exists():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    trace = ExecutionTrace(**data)
                    self._in_memory_traces[trace_id] = trace
                    return trace
            except Exception:
                return None
        return None

execution_trace_logger = ExecutionTraceLogger()
