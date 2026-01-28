# api/session.py
"""
Session management for the legal research chatbot.
Handles conversation history, session persistence, and context tracking.
"""

import uuid
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from pathlib import Path
import os


@dataclass
class ChatMessage:
    """Represents a single chat message."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChatMessage":
        return cls(**data)


@dataclass
class ChatSession:
    """Represents a chat session with history and metadata."""
    session_id: str
    title: str = "New Chat"
    messages: List[ChatMessage] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_message(self, role: str, content: str, metadata: Dict[str, Any] = None) -> ChatMessage:
        """Add a message to the session."""
        message = ChatMessage(
            role=role,
            content=content,
            metadata=metadata or {}
        )
        self.messages.append(message)
        self.updated_at = datetime.now().isoformat()

        # Auto-generate title from first user message
        if self.title == "New Chat" and role == "user":
            self.title = content[:50] + "..." if len(content) > 50 else content

        return message

    def get_history(self, last_n: int = None) -> List[Dict[str, str]]:
        """Get conversation history in LangChain format."""
        messages = self.messages if last_n is None else self.messages[-last_n:]
        return [{"role": m.role, "content": m.content} for m in messages]

    def get_context_summary(self) -> str:
        """Get a summary of the conversation context."""
        if not self.messages:
            return "No previous context."

        user_messages = [m.content for m in self.messages if m.role == "user"]
        if not user_messages:
            return "No previous queries."

        return f"Previous queries in this session: {'; '.join(user_messages[-3:])}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "title": self.title,
            "messages": [m.to_dict() for m in self.messages],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChatSession":
        messages = [ChatMessage.from_dict(m) for m in data.get("messages", [])]
        return cls(
            session_id=data["session_id"],
            title=data.get("title", "New Chat"),
            messages=messages,
            created_at=data.get("created_at", datetime.now().isoformat()),
            updated_at=data.get("updated_at", datetime.now().isoformat()),
            metadata=data.get("metadata", {})
        )

    def clear(self):
        """Clear all messages but keep the session."""
        self.messages = []
        self.title = "New Chat"
        self.updated_at = datetime.now().isoformat()


class SessionManager:
    """
    Manages multiple chat sessions with persistence.
    """

    def __init__(self, storage_dir: str = None):
        """
        Initialize session manager.

        Args:
            storage_dir: Directory to store session files. If None, sessions are in-memory only.
        """
        self.sessions: Dict[str, ChatSession] = {}
        self.storage_dir = Path(storage_dir) if storage_dir else None

        if self.storage_dir:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            self._load_sessions()

    def create_session(self, session_id: str = None, title: str = "New Chat") -> ChatSession:
        """Create a new chat session."""
        session_id = session_id or str(uuid.uuid4())
        session = ChatSession(session_id=session_id, title=title)
        self.sessions[session_id] = session

        if self.storage_dir:
            self._save_session(session)

        return session

    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """Get a session by ID."""
        return self.sessions.get(session_id)

    def get_or_create_session(self, session_id: str = None) -> ChatSession:
        """Get existing session or create a new one."""
        if session_id and session_id in self.sessions:
            return self.sessions[session_id]
        return self.create_session(session_id)

    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        if session_id in self.sessions:
            del self.sessions[session_id]

            if self.storage_dir:
                session_file = self.storage_dir / f"{session_id}.json"
                if session_file.exists():
                    session_file.unlink()

            return True
        return False

    def list_sessions(self, limit: int = 20) -> List[Dict[str, Any]]:
        """List all sessions with basic info."""
        sessions = sorted(
            self.sessions.values(),
            key=lambda s: s.updated_at,
            reverse=True
        )[:limit]

        return [
            {
                "session_id": s.session_id,
                "title": s.title,
                "message_count": len(s.messages),
                "created_at": s.created_at,
                "updated_at": s.updated_at
            }
            for s in sessions
        ]

    def _save_session(self, session: ChatSession):
        """Save a session to disk."""
        if not self.storage_dir:
            return

        session_file = self.storage_dir / f"{session.session_id}.json"
        with open(session_file, "w", encoding="utf-8") as f:
            json.dump(session.to_dict(), f, indent=2, ensure_ascii=False)

    def _load_sessions(self):
        """Load all sessions from disk."""
        if not self.storage_dir:
            return

        for session_file in self.storage_dir.glob("*.json"):
            try:
                with open(session_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    session = ChatSession.from_dict(data)
                    self.sessions[session.session_id] = session
            except Exception as e:
                print(f"Error loading session {session_file}: {e}")

    def save_all(self):
        """Save all sessions to disk."""
        if not self.storage_dir:
            return

        for session in self.sessions.values():
            self._save_session(session)

    def add_message_to_session(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Dict[str, Any] = None
    ) -> ChatMessage:
        """Add a message to a specific session."""
        session = self.get_or_create_session(session_id)
        message = session.add_message(role, content, metadata)

        if self.storage_dir:
            self._save_session(session)

        return message
