from typing import TypedDict, List, Dict, Any, Optional

class AgentState(TypedDict):
    """
    Shared state container passed through each node in the LangGraph workflow.
    """
    query: str
    image_ids: List[str]
    parameters: Dict[str, Any]
    
    # Node 1: Intent Classification Output
    intent: Optional[str]
    intent_confidence: float
    intent_explanation: str
    
    # Node 2: Validation
    is_valid: bool
    validation_error: Optional[str]
    
    # Node 3: Metadata Reader Output
    metadata: Dict[str, Any]
    
    # Node 4: Dispatched Tool / Model Outputs
    models_dispatched: List[str]
    tool_outputs: Dict[str, Any]
    selected_model: Optional[str]
    cdchat_latency: float
    
    # Node 5: Spatial Analysis Output
    spatial_analysis_results: Dict[str, Any]
    
    # Node 6: Evidence Fusion Output
    fused_evidence: Dict[str, Any]
    
    # Node 7: Final Answer Output
    final_answer: str
    overall_confidence: float
    
    # Execution Tracking
    trace_id: Optional[str]
    start_time: float
    end_time: float
    total_latency: float
    execution_logs: List[str]
    pipeline_errors: List[str]
