from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime


class CreateSessionRequest(BaseModel):
    selected_costing: str
    mode: str
    topic_code: Optional[str] = None
    training_code: Optional[str] = None


class CreateSessionResponse(BaseModel):
    session_key: str
    mode: str


class ChatRequest(BaseModel):
    session_key: str
    message: str
    mode: str
    topic_code: Optional[str] = None
    training_code: Optional[str] = None
    conversation_id: Optional[str] = None  # links messages to a conversation


class ChatResponse(BaseModel):
    answer: str
    sources: List[str]


class SessionSummary(BaseModel):
    session_key: str
    selected_costing: Optional[str]
    mode: str
    topic_code: Optional[str]
    training_code: Optional[str]
    created_at: datetime
    last_message: Optional[str] = None


class MessageOut(BaseModel):
    role: str
    message: str
    sources: Optional[List[str]]
    created_at: datetime


class CreateConversationRequest(BaseModel):
    session_key: str
    title: Optional[str] = "New Conversation"


class ConversationOut(BaseModel):
    id: str
    session_key: str
    title: str
    mode: str
    selected_costing: Optional[str]
    topic_code: Optional[str]
    training_code: Optional[str]
    created_at: datetime
    updated_at: datetime
    last_message: Optional[str] = None


class ConversationMessagesResponse(BaseModel):
    conversation_id: str
    title: str
    messages: List[MessageOut]