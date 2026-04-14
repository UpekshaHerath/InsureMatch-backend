import uuid
import logging
from fastapi import APIRouter, HTTPException

from app.models.schemas import ChatRequest, ChatResponse
from app.core.rag.chain import chat, clear_session

router = APIRouter(prefix="/api/chat", tags=["Chat"])
logger = logging.getLogger(__name__)


@router.post("", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Conversational Q&A about insurance policies.

    - Provide a `session_id` to continue a conversation (use the one from /api/recommend).
    - Optionally include `user_profile` so the advisor is aware of your context.
    - The chat maintains conversation history per session (last 6 turns).

    Example questions:
    - "What are the exclusions for this policy?"
    - "How do I make a claim?"
    - "What is the waiting period for critical illness?"
    - "Can I add a rider to this plan?"
    """
    session_id = request.session_id or str(uuid.uuid4())

    try:
        response_text, sources = await chat(
            session_id=session_id,
            message=request.message,
            user_profile=request.user_profile,
        )
        return ChatResponse(
            session_id=session_id,
            response=response_text,
            sources=sources,
        )
    except Exception as e:
        logger.error(f"Chat error [session={session_id}]: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")


@router.delete("/{session_id}")
async def end_session(session_id: str):
    """Clear conversation history for a session."""
    clear_session(session_id)
    return {"message": f"Session '{session_id}' cleared."}


@router.post("/new")
async def new_session():
    """Generate a fresh session ID for a new conversation."""
    return {"session_id": str(uuid.uuid4())}
