"""
The backend contract — the single extension point for wiring Forage to a corpus.

A backend is anything with two methods. Implement them and pass the object as
``SearchAgent(search_backend=...)``; the agent never imports a concrete backend.

    class MyBackend:
        def search(self, query: str, top_k: int = 10) -> list[tuple[str, str, float]]:
            '''Semantic / hybrid retrieval. Returns (doc_id, full_text, score 0..1),
            best first.'''

        def grep(self, pattern: str, top_k: int = 10) -> list[tuple[str, str, float]]:
            '''Exact / regex matching — the strings semantic search misses.
            Same return shape.'''

Reference implementations: `forage.backends.memory.InMemorySearchBackend`
(dependency-free, used by the test suite) and
`forage.backends.chroma.ChromaSearchBackend` (ChromaDB, ``pip install forage-agent[chroma]``).
"""

from typing import List, Protocol, Tuple, runtime_checkable


@runtime_checkable
class SearchBackend(Protocol):
    """Structural interface every Forage backend satisfies."""

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, str, float]]:
        """Return up to top_k (doc_id, full_text, score) hits, best first."""
        ...

    def grep(self, pattern: str, top_k: int = 10) -> List[Tuple[str, str, float]]:
        """Return up to top_k (doc_id, full_text, score) exact/regex matches."""
        ...
