"""Pluggable retrieval backends. Implement `SearchBackend` (two methods) to wire any corpus."""

from .base import SearchBackend
from .memory import InMemorySearchBackend
from .chroma import ChromaSearchBackend

__all__ = ["SearchBackend", "InMemorySearchBackend", "ChromaSearchBackend"]
