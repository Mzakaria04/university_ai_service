import logging
from fastapi import APIRouter, Security, HTTPException, Depends
from fastapi.security.api_key import APIKeyHeader
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession
import tiktoken

from ai_service.config.settings import settings
from ai_service.db.session import AsyncSessionLocal
from ai_service.db.models import AISession, AIMessageEvent, AIExecutionTrace
from ai_service.tools.registry import ToolRegistry, ROLE_TOOL_PERMISSIONS
from ai_service.providers.failover import FailoverProviderOrchestrator

logger = logging.getLogger("ai_service.api.internal.debug")
router = APIRouter()

# API Key dependency setup
api_key_header = APIKeyHeader(name="X-Internal-Key", auto_error=False)

async def verify_internal_key(x_internal_key: str = Security(api_key_header)):
    if not x_internal_key or x_internal_key != settings.INTERNAL_API_KEY:
        logger.warning("Unauthorized debug API access attempt.")
        raise HTTPException(status_code=403, detail="Forbidden: Invalid Internal API Key")
    return x_internal_key

# Async DB dependency setup
async def get_db_session() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session

@router.get("/session/{session_id}/memory", dependencies=[Depends(verify_internal_key)])
async def get_session_memory_debug(session_id: str, db: AsyncSession = Depends(get_db_session)):
    """
    Returns window size, summary presence, and token counts for a session.
    """
    # Verify session exists
    session_query = select(AISession).where(AISession.id == session_id)
    session_res = await db.execute(session_query)
    session_record = session_res.scalar_one_or_none()
    if not session_record:
        raise HTTPException(status_code=404, detail="Session not found")

    # Fetch unsummarized messages
    msg_query = (
        select(AIMessageEvent)
        .where(AIMessageEvent.session_id == session_id)
        .where(AIMessageEvent.is_summarized == False)
        .where(AIMessageEvent.message_type != "summary")
        .order_by(AIMessageEvent.sequence_number.asc())
    )
    msg_res = await db.execute(msg_query)
    unsummarized = msg_res.scalars().all()

    # Calculate tokens
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
    except Exception:
        encoding = tiktoken.encoding_for_model("gpt-4")

    total_tokens = sum(len(encoding.encode(m.content or "")) for m in unsummarized)

    return {
        "session_id": session_id,
        "summary_present": session_record.summary_text is not None,
        "summary_text": session_record.summary_text,
        "unsummarized_message_count": len(unsummarized),
        "estimated_unsummarized_tokens": total_tokens,
        "short_term_window_messages": [
            {
                "id": m.id,
                "role": m.role,
                "message_type": m.message_type,
                "sequence_number": m.sequence_number,
                "token_estimate": len(encoding.encode(m.content or ""))
            } for m in unsummarized
        ]
    }

@router.get("/session/{session_id}/trace", dependencies=[Depends(verify_internal_key)])
async def get_session_traces_debug(session_id: str, last_n: int = 5, db: AsyncSession = Depends(get_db_session)):
    """
    Returns recent ai_execution_traces rows for a session.
    """
    query = (
        select(AIExecutionTrace)
        .where(AIExecutionTrace.session_id == session_id)
        .order_by(desc(AIExecutionTrace.created_at))
        .limit(last_n)
    )
    result = await db.execute(query)
    traces = result.scalars().all()

    return [
        {
            "id": t.id,
            "request_id": t.request_id,
            "provider_used": t.provider_used,
            "model_used": t.model_used,
            "provider_fallback": t.provider_fallback,
            "prompt_tokens": t.prompt_tokens,
            "completion_tokens": t.completion_tokens,
            "total_tokens": t.total_tokens,
            "tool_calls_count": t.tool_calls_count,
            "tools_used": t.tools_used,
            "latency_ms": t.latency_ms,
            "success": t.success,
            "error_type": t.error_type,
            "created_at": t.created_at
        } for t in traces
    ]

@router.get("/tools", dependencies=[Depends(verify_internal_key)])
async def get_tools_debug():
    """
    Lists all registered tools with their schemas and allowed roles.
    """
    tools = []
    for name, tool_def in ToolRegistry._tools.items():
        tools.append({
            "name": name,
            "description": tool_def.description,
            "domain": tool_def.domain.value if hasattr(tool_def.domain, "value") else str(tool_def.domain),
            "allowed_roles": [role.value if hasattr(role, "value") else str(role) for role in tool_def.allowed_roles],
            "schema": tool_def.to_llm_schema(),
            "timeout_seconds": tool_def.timeout_seconds,
            "max_retries": tool_def.max_retries
        })
    return {
        "registered_tools_count": len(tools),
        "tools": tools,
        "role_permissions": {
            role.value if hasattr(role, "value") else str(role): list(perms)
            for role, perms in ROLE_TOOL_PERMISSIONS.items()
        }
    }

@router.get("/providers/health", dependencies=[Depends(verify_internal_key)])
async def get_providers_health_debug(db: AsyncSession = Depends(get_db_session)):
    """
    Returns circuit breaker states and recent execution latencies.
    """
    cb = FailoverProviderOrchestrator._shared_circuit_breaker
    
    # Query database for recent execution latencies
    query = (
        select(AIExecutionTrace)
        .order_by(desc(AIExecutionTrace.created_at))
        .limit(5)
    )
    result = await db.execute(query)
    traces = result.scalars().all()
    
    recent_latencies = [
        {
            "session_id": t.session_id,
            "request_id": t.request_id,
            "provider": t.provider_used,
            "model": t.model_used,
            "fallback": t.provider_fallback,
            "latency_ms": t.latency_ms,
            "success": t.success,
            "created_at": t.created_at
        } for t in traces
    ]

    return {
        "circuit_breaker": {
            "state": cb.state.value if hasattr(cb.state, "value") else str(cb.state),
            "failures": cb.failures,
            "last_failure_time": cb.last_failure_time,
            "failure_threshold": cb.failure_threshold,
            "recovery_timeout_seconds": cb.recovery_timeout
        },
        "recent_latencies": recent_latencies
    }

@router.get("/session/{session_id}/messages", dependencies=[Depends(verify_internal_key)])
async def get_session_messages_debug(
    session_id: str, 
    include_tools: bool = True, 
    db: AsyncSession = Depends(get_db_session)
):
    """
    Returns full ai_message_events history for a session.
    """
    query = (
        select(AIMessageEvent)
        .where(AIMessageEvent.session_id == session_id)
        .order_by(AIMessageEvent.sequence_number.asc())
    )
    result = await db.execute(query)
    messages = result.scalars().all()

    formatted_messages = []
    for m in messages:
        # If include_tools is False, filter out tool-related events
        if not include_tools and (m.role == "tool" or m.message_type in ("tool_call", "tool_result")):
            continue
            
        formatted_messages.append({
            "id": m.id,
            "role": m.role,
            "message_type": m.message_type,
            "content": m.content,
            "tool_call_id": m.tool_call_id,
            "tool_name": m.tool_name,
            "is_summarized": m.is_summarized,
            "sequence_number": m.sequence_number,
            "created_at": m.created_at,
            "metadata_json": m.metadata_json
        })

    return formatted_messages
