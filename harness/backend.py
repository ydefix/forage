"""
Minimal ChromaDB search backend with in-memory document store.
For demo/testing before wiring to a real corpus.
"""

from typing import List, Tuple, Optional


class InMemorySearchBackend:
    """
    Simple in-memory search backend for testing the harness.
    Replace with ChromaDB for production use.
    """
    
    def __init__(self, documents: List[Tuple[str, str]] = None):
        """
        Args:
            documents: List of (doc_id, text) tuples to index.
        """
        self._docs: List[Tuple[str, str]] = []
        if documents:
            for doc_id, text in documents:
                self._docs.append((doc_id, text))
    
    def add(self, doc_id: str, text: str):
        """Add a document to the index."""
        self._docs.append((doc_id, text))
    
    def load_from_markdown(self, path: str):
        """Load documents from a markdown file, splitting by headings."""
        with open(path) as f:
            content = f.read()
        
        # Split by ## headings
        sections = content.split("\n## ")
        for i, section in enumerate(sections):
            if not section.strip():
                continue
            
            # First line is the heading
            lines = section.split("\n", 1)
            heading = lines[0].strip("# ").strip()
            body = lines[1] if len(lines) > 1 else ""
            
            doc_id = f"doc_{i:04d}"
            self._docs.append((doc_id, f"# {heading}\n{body}"))
        
        print(f"Loaded {len(self._docs)} documents from {path}")
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, str, float]]:
        """
        Simple keyword-based search. Returns (doc_id, text, score).
        """
        query_terms = query.lower().split()
        scored = []
        for doc_id, text in self._docs:
            text_lower = text.lower()
            score = sum(1 for term in query_terms if term in text_lower)
            if score > 0:
                scored.append((doc_id, text, score / len(query_terms)))
        
        scored.sort(key=lambda x: -x[2])
        return scored[:top_k]
    
    def grep(self, pattern: str, top_k: int = 10) -> List[Tuple[str, str, float]]:
        """
        Simple substring matching. Returns (doc_id, text, score).
        """
        import re
        scored = []
        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error:
            # Fall back to substring
            pattern_lower = pattern.lower()
            for doc_id, text in self._docs:
                if pattern_lower in text.lower():
                    scored.append((doc_id, text, 1.0))
            return scored[:top_k]
        
        for doc_id, text in self._docs:
            matches = len(regex.findall(text))
            if matches > 0:
                scored.append((doc_id, text, min(matches / 10, 1.0)))
        
        scored.sort(key=lambda x: -x[2])
        return scored[:top_k]
