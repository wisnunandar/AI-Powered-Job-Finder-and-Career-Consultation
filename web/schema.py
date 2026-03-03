from typing import Literal

from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: Literal["user", "ai"]
    content: str


class ChatRequest(BaseModel):
    history: list[ChatMessage]
    message: ChatMessage
    session_id: str


class ChatResponse(BaseModel):
    message: ChatMessage
    agent_used: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
