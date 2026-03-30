import uuid
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy.sql import func
from app.core.database import get_db
from app.models.knowledge import Topic, TrainingModule
from app.models.chat import ChatSession, ChatMessage, Conversation
from app.schemas.chat import (
    CreateSessionRequest,
    CreateSessionResponse,
    ChatRequest,
    ChatResponse,
    SessionSummary,
    MessageOut,
    CreateConversationRequest,
    ConversationOut,
    ConversationMessagesResponse,
)
from app.services.chat_service import ChatGenerationError, generate_answer

router = APIRouter()


@router.get("/startup-options")
def get_startup_options():
    return [
        {
            "id": "overview",
            "label": "Have a general overview",
            "description": "Get a broad overview of the costing knowledge base."
        },
        {
            "id": "topic",
            "label": "Engage a discussion about a specific topic",
            "description": "Choose a topic and explore it in depth."
        },
        {
            "id": "training",
            "label": "Get some training",
            "description": "Follow a training-oriented conversation."
        }
    ]


@router.get("/costing-options")
def get_costing_options():
    return [
        {
            "id": "product_costing",
            "label": "Product Costing"
        }
    ]


@router.get("/topics")
def get_topics(db: Session = Depends(get_db)):
    items = db.query(Topic).order_by(Topic.code.asc()).all()
    return [
        {"code": t.code, "title": t.title, "summary": t.summary}
        for t in items
    ]


@router.get("/training-modules")
def get_training_modules(db: Session = Depends(get_db)):
    items = db.query(TrainingModule).order_by(TrainingModule.code.asc()).all()
    return [
        {"code": t.code, "title": t.title, "objective": t.objective}
        for t in items
    ]


@router.post("/sessions", response_model=CreateSessionResponse)
def create_session(payload: CreateSessionRequest, db: Session = Depends(get_db)):
    session_key = str(uuid.uuid4())
    session = ChatSession(
        session_key=session_key,
        selected_costing=payload.selected_costing,
        mode=payload.mode,
        topic_code=payload.topic_code,
        training_code=payload.training_code
    )
    db.add(session)
    db.commit()
    return CreateSessionResponse(session_key=session_key, mode=payload.mode)


from fastapi.responses import StreamingResponse
from app.services.chat_service import generate_answer

@router.post("/chat")
async def chat(payload: ChatRequest, db: Session = Depends(get_db)):
    session = db.query(ChatSession).filter(ChatSession.session_key == payload.session_key).first()

    if session is None:
        raise HTTPException(status_code=404, detail="Chat session not found.")

    # Resolve conversation_id
    conversation_id = None
    if payload.conversation_id:
        import uuid as _uuid
        try:
            cid = _uuid.UUID(str(payload.conversation_id))
        except ValueError:
            raise HTTPException(status_code=422, detail="Invalid conversation_id format.")

        conv = db.query(Conversation).filter(Conversation.id == cid).first()
        if conv:
            conversation_id = conv.id
            conv.updated_at = datetime.utcnow()
        else:
            print(f"[chat] WARNING: conversation {payload.conversation_id} not found in DB")
    else:
        conv = (
            db.query(Conversation)
            .filter(Conversation.session_id == session.id)
            .order_by(Conversation.updated_at.desc())
            .first()
        )
        if conv:
            conversation_id = conv.id
            conv.updated_at = datetime.utcnow()

    # Save User Message
    db.add(ChatMessage(
        session_id=session.id,
        conversation_id=conversation_id,
        role="user",
        message=payload.message,
        sources=[]
    ))
    db.commit()

    async def response_generator():
        full_answer = ""
        async for chunk in generate_answer(
            db=db,
            message=payload.message,
            mode=session.mode,
            topic_code=session.topic_code,
            training_code=session.training_code,
        ):
            full_answer += chunk
            yield chunk

        # Save Assistant Message at the end of the stream
        db.add(ChatMessage(
            session_id=session.id,
            conversation_id=conversation_id,
            role="assistant",
            message=full_answer,
            sources=[] # We hid sources as requested
        ))
        db.commit()

    return StreamingResponse(response_generator(), media_type="text/plain")



# ── Conversations ──────────────────────────────────────────────────────────────

@router.post("/conversations", response_model=ConversationOut)
def create_conversation(payload: CreateConversationRequest, db: Session = Depends(get_db)):
    """Create a new named conversation linked to a session."""
    session = db.query(ChatSession).filter(ChatSession.session_key == payload.session_key).first()
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found.")

    conv = Conversation(
        session_id=session.id,
        title=payload.title or "New Conversation",
    )
    db.add(conv)
    db.commit()
    db.refresh(conv)

    return ConversationOut(
        id=str(conv.id),
        session_key=session.session_key,
        title=conv.title,
        mode=session.mode,
        selected_costing=session.selected_costing,
        topic_code=session.topic_code,
        training_code=session.training_code,
        created_at=conv.created_at,
        updated_at=conv.updated_at,
        last_message=None,
    )


@router.get("/conversations", response_model=list[ConversationOut])
def list_conversations(db: Session = Depends(get_db)):
    """List all conversations, newest first, with last message preview."""
    conversations = (
        db.query(Conversation)
        .order_by(Conversation.updated_at.desc())
        .all()
    )

    result = []
    for conv in conversations:
        session = db.query(ChatSession).filter(ChatSession.id == conv.session_id).first()
        last = (
            db.query(ChatMessage)
            .filter(ChatMessage.conversation_id == conv.id)
            .order_by(ChatMessage.created_at.desc())
            .first()
        )
        result.append(
            ConversationOut(
                id=str(conv.id),
                session_key=session.session_key if session else "",
                title=conv.title,
                mode=session.mode if session else "",
                selected_costing=session.selected_costing if session else None,
                topic_code=session.topic_code if session else None,
                training_code=session.training_code if session else None,
                created_at=conv.created_at,
                updated_at=conv.updated_at,
                last_message=last.message[:80] if last else None,
            )
        )

    return result


@router.get("/conversations/{conversation_id}/messages", response_model=ConversationMessagesResponse)
def get_conversation_messages(conversation_id: str, db: Session = Depends(get_db)):
    """Load the full message history for a conversation."""
    conv = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")

    messages = (
        db.query(ChatMessage)
        .filter(ChatMessage.conversation_id == conv.id)
        .order_by(ChatMessage.created_at.asc())
        .all()
    )

    return ConversationMessagesResponse(
        conversation_id=str(conv.id),
        title=conv.title,
        messages=[
            MessageOut(
                role=m.role,
                message=m.message,
                sources=m.sources or [],
                created_at=m.created_at,
            )
            for m in messages
        ],
    )


@router.patch("/conversations/{conversation_id}/title")
def rename_conversation(conversation_id: str, payload: dict, db: Session = Depends(get_db)):
    """Rename a conversation."""
    conv = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")

    new_title = payload.get("title", "").strip()
    if not new_title:
        raise HTTPException(status_code=422, detail="Title cannot be empty.")

    conv.title = new_title
    db.commit()
    return {"id": conversation_id, "title": conv.title}


@router.delete("/conversations/{conversation_id}", status_code=204)
def delete_conversation(conversation_id: str, db: Session = Depends(get_db)):
    """Delete a conversation and all its messages."""
    conv = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")

    db.delete(conv)
    db.commit()
