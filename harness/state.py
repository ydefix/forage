"""
Search harness state management.
Adapted from harness-1 (Pengcheng Jiang et al., 2026).
Simplified for Qwen3/Gemma base models.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
import hashlib
import time


@dataclass
class Document:
    """A document chunk from the retrieval corpus."""
    doc_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    snippet: str = ""  # truncated preview
    
    def __hash__(self):
        return hash(self.doc_id)
    
    def __eq__(self, other):
        return isinstance(other, Document) and self.doc_id == other.doc_id


@dataclass 
class SearchResult:
    """Result of a single search operation."""
    query: str
    timestamp: float
    doc_ids: List[str] = field(default_factory=list)
    hit_count: int = 0


@dataclass
class VerificationRecord:
    """Record of a claim verification."""
    doc_id: str
    claim: str
    supports: bool
    reasoning: str = ""


@dataclass
class WorkingMemory:
    """
    Stateful working memory for the search agent.
    All state is externalized - recoverable from trajectory.
    
    Tracks:
    - curated_docs: documents selected as evidence
    - candidate_pool: all documents found
    - search_history: past queries and their results
    - evidence_links: connections between documents
    - verification_records: fact-checking results
    - token_budget: remaining context budget
    """
    
    curated_docs: Dict[str, Document] = field(default_factory=dict)
    candidate_pool: Dict[str, Document] = field(default_factory=dict)
    search_history: List[SearchResult] = field(default_factory=list)
    evidence_links: Dict[str, List[str]] = field(default_factory=dict)
    verification_records: List[VerificationRecord] = field(default_factory=list)
    
    # Constants
    max_curated_docs: int = 30
    max_pool_docs: int = 200
    doc_snippet_chars: int = 120
    
    def add_candidate(self, doc: Document) -> bool:
        """Add a document to the candidate pool. Returns True if new."""
        if doc.doc_id in self.candidate_pool:
            return False
        if len(self.candidate_pool) >= self.max_pool_docs:
            return False
        doc.snippet = doc.text[:self.doc_snippet_chars].replace("\n", " ")
        self.candidate_pool[doc.doc_id] = doc
        return True
    
    def curate(self, add_ids: List[str], remove_ids: List[str] = None):
        """Move documents between candidate pool and curated set."""
        for doc_id in add_ids:
            if len(self.curated_docs) >= self.max_curated_docs:
                break
            if doc_id in self.candidate_pool and doc_id not in self.curated_docs:
                self.curated_docs[doc_id] = self.candidate_pool[doc_id]
        
        for doc_id in (remove_ids or []):
            self.curated_docs.pop(doc_id, None)
    
    def record_search(self, query: str, doc_ids: List[str], hit_count: int):
        """Record a search operation in history."""
        self.search_history.append(SearchResult(
            query=query,
            timestamp=time.time(),
            doc_ids=doc_ids,
            hit_count=hit_count,
        ))
    
    def get_past_queries(self) -> List[str]:
        """Get list of past search queries (for avoiding duplicates)."""
        return [s.query for s in self.search_history]
    
    def get_doc(self, doc_id: str) -> Optional[Document]:
        """Get document from either curated or candidate pool."""
        return self.curated_docs.get(doc_id) or self.candidate_pool.get(doc_id)
    
    def link_evidence(self, source_id: str, target_id: str):
        """Create an evidence link between documents."""
        if source_id not in self.evidence_links:
            self.evidence_links[source_id] = []
        if target_id not in self.evidence_links[source_id]:
            self.evidence_links[source_id].append(target_id)
    
    def add_verification(self, doc_id: str, claim: str, supports: bool, reasoning: str = ""):
        """Record a claim verification."""
        self.verification_records.append(VerificationRecord(
            doc_id=doc_id,
            claim=claim,
            supports=supports,
            reasoning=reasoning,
        ))
    
    def summarize(self) -> str:
        """Generate a text summary of current search state for the LLM context."""
        lines = []
        lines.append(f"[Working Memory]")
        lines.append(f"  Curated docs: {len(self.curated_docs)}/{self.max_curated_docs}")
        lines.append(f"  Pool docs: {len(self.candidate_pool)}")
        lines.append(f"  Searches performed: {len(self.search_history)}")
        
        if self.curated_docs:
            lines.append(f"\n  === Curated Documents ===")
            for i, (doc_id, doc) in enumerate(self.curated_docs.items(), 1):
                lines.append(f"  [{i}] {doc_id}: {doc.snippet}")
        
        if self.search_history:
            lines.append(f"\n  === Recent Searches ===")
            for i, sr in enumerate(self.search_history[-5:], 1):
                lines.append(f"  [{i}] query='{sr.query}' -> {sr.hit_count} hits")
        
        return "\n".join(lines)
    
    def clear(self):
        """Reset all state."""
        self.curated_docs.clear()
        self.candidate_pool.clear()
        self.search_history.clear()
        self.evidence_links.clear()
        self.verification_records.clear()


@dataclass
class TurnContext:
    """Assembled context for a single inference turn."""
    system_prompt: str
    working_memory_summary: str
    recent_observations: List[str]  # last K observations
    action_history: List[str]       # last K actions
    query: str                      # the user's question
    
    def to_messages(self) -> List[Dict[str, str]]:
        """Convert to chat messages for Qwen3/Gemma."""
        messages = []
        
        # System prompt
        full_system = self.system_prompt + "\n\n" + self.working_memory_summary
        messages.append({"role": "system", "content": full_system})
        
        # Recent history
        for obs, act in zip(self.recent_observations, self.action_history):
            messages.append({"role": "assistant", "content": act})
            messages.append({"role": "user", "content": obs})
        
        return messages
