# api/__init__.py
"""
API Layer for Legal Research Chatbot

Provides all necessary APIs for building a chatbot interface:
- Session management
- Query processing
- Suggestions
- Export functionality
"""

from .session import SessionManager, ChatSession
from .chatbot import LegalChatbot, ChatResponse
from .suggestions import QuerySuggestions, QUICK_ACTIONS
from .export import ExportManager

__all__ = [
    "SessionManager",
    "ChatSession",
    "LegalChatbot",
    "ChatResponse",
    "QuerySuggestions",
    "QUICK_ACTIONS",
    "ExportManager",
]
