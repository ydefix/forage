"""
Qwen3.6 search agent — stateful search harness with tool loop.
Connects to oMLX endpoint on Mac Studio.

Adapted from harness-1 (Pengcheng Jiang et al., 2026).
"""

import json
import os
import time
import re
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import requests

from .state import WorkingMemory, Document, TurnContext
from .tools import ALL_TOOLS


# ─── System Prompt (adapted from harness-1 ultra_core.py) ────────────────────

SYSTEM_PROMPT = """You are a search agent. Your job: find and curate the most relevant documents from a retrieval corpus to answer a user's question. You do NOT answer questions yourself — you find evidence and curate it.

<rules>
1. SEARCH → CURATE rhythm: After EVERY search, call curate() to add ALL plausibly relevant documents. Never do two searches in a row without curating between them.
2. CURATE AGGRESSIVELY: Add borderline docs. Over-curation is better than missing evidence. Target 3-8 adds per curate call.
3. SEARCH STRATEGY: Decompose complex queries into facets. Use different angles each search. Keep queries short (5-12 words). Never repeat queries from your search history.
4. GREP for specifics: Use grep_corpus for names, dates, numbers, codes. grep finds what semantic search misses.
5. READ DOCUMENTS: If a snippet looks promising, call read_document() to see the full text.
6. BUDGET AWARENESS: You have limited context. Prioritize the most relevant documents.
7. VERIFY KEY CLAIMS: Before ending, call verify() on the claims your evidence must carry — cite the doc_ids and record whether they support the claim. Verification records travel with the final set.
8. END when done: Call end_search() when you have sufficient curated evidence. Max 35 turns.
9. REASON FIRST: Before each action, explain your reasoning. What do you know? What should you search next? What evidence is still missing?
</rules>

<output_format>
Always reason before acting. Then output a tool call.
Final answer goes through end_search(reasoning="...").
</output_format>"""


@dataclass
class SearchAgent:
    """
    Stateful search agent powered by oMLX Qwen3.6.
    
    Usage:
        agent = SearchAgent(base_url="http://<remote-host>:9393", model="Qwen3.6-27B-OptiQ-4bit")
        result = agent.search("What is the impact of climate change on agriculture?")
        # result.curated_docs has the final evidence set
    """
    
    base_url: str = field(default_factory=lambda: os.environ.get("FORAGE_OMLX_URL", "http://localhost:9393"))
    model: str = "Qwen3.6-27B-OptiQ-4bit"
    max_turns: int = 35
    temperature: float = 0.7
    search_backend: Any = None  # ChromaDB client, injected
    
    # Internal state
    _memory: WorkingMemory = field(default_factory=WorkingMemory)
    _history: List[Dict[str, str]] = field(default_factory=list)
    _turn_count: int = 0
    
    def reset(self):
        """Reset agent state for a new search."""
        self._memory.clear()
        self._history = []
        self._turn_count = 0
    
    def search(self, query: str) -> WorkingMemory:
        """
        Run a complete search for a query. Returns the working memory 
        with curated evidence.
        """
        self.reset()
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"<query>\n{query}\n</query>\n\nBegin your search. Decompose the query, plan your search strategy, and start searching."},
        ]
        
        while self._turn_count < self.max_turns:
            self._turn_count += 1
            
            # Inject working memory summary before each turn
            memory_summary = self._memory.summarize()
            if self._turn_count > 1:
                # Replace the system message with updated memory
                messages[0] = {
                    "role": "system", 
                    "content": SYSTEM_PROMPT + "\n\n" + memory_summary
                }
            
            # Call the model
            response = self._call_llm(messages)
            if response is None:
                break
            
            assistant_msg = response.get("content", "")
            tool_calls = response.get("tool_calls", [])
            
            if not tool_calls and not assistant_msg:
                break
            
            # Record the assistant turn
            turn_record = {"role": "assistant", "content": assistant_msg}
            if tool_calls:
                turn_record["tool_calls"] = tool_calls
            messages.append(turn_record)
            
            # Execute tool calls
            for tc in tool_calls:
                func_name = tc["function"]["name"]
                func_args = json.loads(tc["function"]["arguments"])
                
                result = self._execute_tool(func_name, func_args)
                
                # Add tool result as a user message (tool role)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id", ""),
                    "content": result,
                })
                
                # Check for end_search
                if func_name == "end_search":
                    return self._memory
            
            # Nudge: if too many turns without curation
            if self._turn_count % 3 == 0 and len(self._memory.curated_docs) == 0:
                messages.append({
                    "role": "user",
                    "content": "[NUDGE] You have searched but not curated. Review your search results and call curate() to add relevant documents."
                })
        
        return self._memory
    
    def _call_llm(self, messages: List[Dict]) -> Optional[Dict]:
        """Call oMLX Qwen3.6 endpoint with tool support."""
        payload = {
            "model": self.model,
            "messages": messages,
            "tools": ALL_TOOLS,
            "max_tokens": 2048,
            "temperature": self.temperature,
            "stream": False,
        }
        
        try:
            resp = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            
            choice = data["choices"][0]
            msg = choice["message"]
            
            return {
                "content": msg.get("content", ""),
                "tool_calls": msg.get("tool_calls", []),
                "finish_reason": choice.get("finish_reason", ""),
            }
        except Exception as e:
            print(f"[ERROR] LLM call failed: {e}")
            return None
    
    def _execute_tool(self, name: str, args: Dict) -> str:
        """Execute a tool and return the result as a string."""
        
        if name == "search_corpus":
            return self._tool_search(query=args["query"])
        
        elif name == "fan_out_search":
            results = []
            for q in args.get("queries", []):
                r = self._tool_search(query=q)
                results.append(f"Query '{q}':\n{r}")
            return "\n\n".join(results)
        
        elif name == "grep_corpus":
            return self._tool_grep(pattern=args["pattern"])
        
        elif name == "read_document":
            return self._tool_read(doc_id=args["doc_id"])
        
        elif name == "curate":
            add_ids = args.get("add_ids", [])
            remove_ids = args.get("remove_ids", [])
            self._memory.curate(add_ids, remove_ids)
            return (f"Curated: added {len(add_ids)}, removed {len(remove_ids)}. "
                    f"Total curated: {len(self._memory.curated_docs)}/{self._memory.max_curated_docs}")
        
        elif name == "verify":
            doc_ids = args.get("doc_ids", [])
            claim = args.get("claim", "")
            supports = bool(args.get("supports", False))
            reasoning = args.get("reasoning", "")
            known, missing = [], []
            for doc_id in doc_ids:
                if self._memory.get_doc(doc_id) is None:
                    missing.append(doc_id)
                    continue
                self._memory.add_verification(doc_id, claim, supports, reasoning)
                known.append(doc_id)
            verdict = "SUPPORTS" if supports else "DOES NOT SUPPORT"
            result = (f"Verified: {len(known)} doc(s) recorded as {verdict} the claim. "
                      f"Total verification records: {len(self._memory.verification_records)}.")
            if missing:
                result += f" Unknown doc_ids skipped: {', '.join(missing)}."
            return result

        elif name == "end_search":
            return f"Search ended. {len(self._memory.curated_docs)} documents curated. Reasoning: {args.get('reasoning', '')}"
        
        else:
            return f"Unknown tool: {name}"
    
    def _tool_search(self, query: str) -> str:
        """Execute a search against the retrieval backend."""
        if self.search_backend is None:
            return "[NO BACKEND] No search backend configured. Set agent.search_backend."
        
        try:
            results = self.search_backend.search(query, top_k=10)
        except Exception as e:
            return f"Search error: {e}"
        
        if not results:
            return "No results found."
        
        # Record in memory and format output
        doc_ids = []
        lines = [f"Search: '{query}' — {len(results)} results:"]
        for i, (doc_id, text, score) in enumerate(results, 1):
            self._memory.add_candidate(Document(doc_id=doc_id, text=text))
            doc_ids.append(doc_id)
            snippet = text[:120].replace("\n", " ")
            lines.append(f"  [{i}] {doc_id} (score={score:.3f}): {snippet}")
        
        self._memory.record_search(query, doc_ids, len(results))
        return "\n".join(lines)
    
    def _tool_grep(self, pattern: str) -> str:
        """Execute regex grep against the corpus."""
        if self.search_backend is None:
            return "[NO BACKEND] No search backend configured."
        
        try:
            results = self.search_backend.grep(pattern, top_k=10)
        except Exception as e:
            return f"Grep error: {e}"
        
        if not results:
            return "No matches found."
        
        doc_ids = []
        lines = [f"Grep: '{pattern}' — {len(results)} matches:"]
        for i, (doc_id, text, _) in enumerate(results, 1):
            self._memory.add_candidate(Document(doc_id=doc_id, text=text))
            doc_ids.append(doc_id)
            snippet = text[:120].replace("\n", " ")
            lines.append(f"  [{i}] {doc_id}: {snippet}")
        
        self._memory.record_search(f"grep:{pattern}", doc_ids, len(results))
        return "\n".join(lines)
    
    def _tool_read(self, doc_id: str) -> str:
        """Read a document's full text."""
        doc = self._memory.get_doc(doc_id)
        if doc is None:
            return f"Document {doc_id} not found in memory."
        
        return f"=== {doc_id} ===\n{doc.text}"
