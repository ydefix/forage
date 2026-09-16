"""Forage — a search agent that curates verified evidence instead of answering."""

from .state import WorkingMemory, Document, TurnContext
from .tools import ALL_TOOLS
from .agent import SearchAgent, SYSTEM_PROMPT
from .backends import SearchBackend, InMemorySearchBackend, ChromaSearchBackend

__all__ = [
    "WorkingMemory", "Document", "TurnContext",
    "ALL_TOOLS", "SearchAgent", "SYSTEM_PROMPT",
    "SearchBackend", "InMemorySearchBackend", "ChromaSearchBackend",
]
