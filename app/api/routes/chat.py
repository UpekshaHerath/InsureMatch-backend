import uuid
import logging
from fastapi import APIRouter, HTTPException, Depends

from app.models.schemas import ChatRequest, ChatResponse
from app.core.rag.chain import chat, clear_session
from app.core.auth.deps import get_current_user, AuthUser
from app.core.db import supabase_client as db

router = APIRouter(prefix="/api/chat", tags=["Chat"])
logger = logging.getLogger(__name__)


@router.post("", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    user: AuthUser = Depends(get_current_user),
):
    """Conversational Q&A. History persisted per (user_id, session_id) in Supabase."""
    session_id = request.session_id or str(uuid.uuid4())

    try:
        response_text, sources = await chat(
            session_id=session_id,
            user_id=user.user_id,
            message=request.message,
            user_profile=request.user_profile,
            recommendation_context=request.recommendation_context,
        )
        return ChatResponse(
            session_id=session_id,
            response=response_text,
            sources=sources,
        )
    except Exception as e:
        logger.error(f"Chat error [user={user.user_id} session={session_id}]: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")


@router.delete("/{session_id}")
async def end_session(
    session_id: str,
    user: AuthUser = Depends(get_current_user),
):
    """Clear conversation history for a session (owner only)."""
    await clear_session(session_id, user.user_id)
    return {"message": f"Session '{session_id}' cleared."}


@router.post("/new")
async def new_session(user: AuthUser = Depends(get_current_user)):
    """Generate a fresh session ID for a new conversation."""
    session_id = str(uuid.uuid4())
    try:
        await db.ensure_session(user.user_id, session_id)
    except Exception as e:
        logger.warning(f"ensure_session failed: {e}")
    return {"session_id": session_id}
