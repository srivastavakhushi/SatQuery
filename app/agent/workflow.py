import time
import uuid
from typing import Dict, Any, List
from langgraph.graph import StateGraph, START, END

from app.agent.state import AgentState
from app.agent.nodes import (
    classify_intent_node,
    validate_images_node,
    read_metadata_node,
    dispatch_tools_node,
    run_spatial_analysis_node,
    evidence_fusion_node,
    generate_answer_node
)
from app.logger import execution_trace_logger

def build_agent_graph():
    """
    Constructs and compiles the LangGraph Agent State Machine.
    """
    builder = StateGraph(AgentState)

    # Register workflow nodes
    builder.add_node("classify_intent", classify_intent_node)
    builder.add_node("validate_images", validate_images_node)
    builder.add_node("read_metadata", read_metadata_node)
    builder.add_node("dispatch_tools", dispatch_tools_node)
    builder.add_node("run_spatial_analysis", run_spatial_analysis_node)
    builder.add_node("evidence_fusion", evidence_fusion_node)
    builder.add_node("generate_answer", generate_answer_node)

    # Define sequential state transition edges
    builder.add_edge(START, "classify_intent")
    builder.add_edge("classify_intent", "validate_images")
    builder.add_edge("validate_images", "read_metadata")
    builder.add_edge("read_metadata", "dispatch_tools")
    builder.add_edge("dispatch_tools", "run_spatial_analysis")
    builder.add_edge("run_spatial_analysis", "evidence_fusion")
    builder.add_edge("evidence_fusion", "generate_answer")
    builder.add_edge("generate_answer", END)

    return builder.compile()

# Compile global agent graph instance
agent_graph = build_agent_graph()

def run_agent_workflow(
    query: str,
    image_ids: List[str],
    parameters: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Executes the Agent State Machine and records an Execution Trace.
    """
    start_time = time.time()
    trace_id = f"trc-{uuid.uuid4().hex[:12]}"

    initial_state: AgentState = {
        "query": query,
        "image_ids": image_ids or [],
        "parameters": parameters or {},
        "intent": None,
        "intent_confidence": 0.0,
        "intent_explanation": "",
        "is_valid": False,
        "validation_error": None,
        "metadata": {},
        "models_dispatched": [],
        "tool_outputs": {},
        "selected_model": None,
        "cdchat_latency": 0.0,
        "spatial_analysis_results": {},
        "fused_evidence": {},
        "final_answer": "",
        "overall_confidence": 0.0,
        "trace_id": trace_id,
        "start_time": start_time,
        "end_time": 0.0,
        "total_latency": 0.0,
        "execution_logs": [],
        "pipeline_errors": [],
    }

    # Run LangGraph State Machine
    final_state = agent_graph.invoke(initial_state)

    end_time = time.time()
    latency = end_time - start_time
    final_state["end_time"] = end_time
    final_state["total_latency"] = latency

    cd_output = (final_state.get("tool_outputs") or {}).get("ChangeDetection") or {}
    trace_parameters = dict(final_state.get("parameters") or {})
    trace_parameters.update({
        "image_ids": final_state.get("image_ids") or [],
        "intent_explanation": final_state.get("intent_explanation"),
        "selected_model": final_state.get("selected_model"),
        "cdchat_latency": final_state.get("cdchat_latency"),
    })

    # Audit Logging via ExecutionTraceLogger
    trace = execution_trace_logger.log_trace(
        query=final_state["query"],
        selected_task=final_state["intent"],
        models_dispatched=final_state["models_dispatched"],
        parameters=trace_parameters,
        latency=latency,
        confidence=final_state["overall_confidence"],
        trace_id=trace_id,
        intent_confidence=final_state.get("intent_confidence"),
        selected_model=final_state.get("selected_model"),
        image_ids=final_state.get("image_ids") or [],
        cdchat_execution_time=final_state.get("cdchat_latency") or 0.0,
        cdchat_result=cd_output or None,
        fusion_result=final_state.get("fused_evidence") or None,
        errors=final_state.get("pipeline_errors") or [],
        execution_logs=final_state.get("execution_logs") or [],
    )

    final_state["execution_trace"] = trace
    return final_state
