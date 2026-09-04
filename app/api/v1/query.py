from fastapi import APIRouter, HTTPException, status
from app.schemas import QueryRequest, QueryResponse
from app.agent import run_agent_workflow
from app.exceptions import QueryPipelineError

router = APIRouter()

@router.post("/query", response_model=QueryResponse, status_code=status.HTTP_200_OK)
async def process_query(request: QueryRequest):
    """
    POST /api/v1/query
    Receives natural language query + optional image IDs, classifies intent,
    executes LangGraph state machine, dispatches tools/models, and returns trace.

    CDChat is invoked only when the classified intent is BI_TEMPORAL_CHANGE.
    """
    if not request.query or not request.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query field cannot be empty."
        )

    try:
        agent_result = run_agent_workflow(
            query=request.query,
            image_ids=request.image_ids,
            parameters=request.parameters
        )

        return QueryResponse(
            status="success",
            query=agent_result["query"],
            intent=agent_result["intent"],
            confidence=agent_result["overall_confidence"],
            answer=agent_result["final_answer"],
            fused_evidence=agent_result.get("fused_evidence", {}),
            models_dispatched=agent_result["models_dispatched"],
            execution_trace=agent_result["execution_trace"]
        )
    except QueryPipelineError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from None
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Agent workflow execution failed."
        )
