"""
Search harness tools with Qwen3 function-calling schemas.
Adapted from harness-1 tool definitions.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass


# ─── Qwen function-calling schemas ───────────────────────────────────────────

FAN_OUT_SEARCH = {
    "type": "function",
    "function": {
        "name": "fan_out_search",
        "description": "Run up to 5 diverse search queries in parallel. Returns combined results from all queries. Best for broad exploration of a topic from multiple angles.",
        "parameters": {
            "type": "object",
            "properties": {
                "queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of search queries (max 5). Each should target a different aspect or angle.",
                    "maxItems": 5,
                }
            },
            "required": ["queries"],
        },
    },
}

SEARCH_CORPUS = {
    "type": "function",
    "function": {
        "name": "search_corpus",
        "description": "Single hybrid semantic + keyword search over the document corpus. Returns top matching document chunks with snippets.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query (5-12 words recommended). Be specific.",
                }
            },
            "required": ["query"],
        },
    },
}

GREP_CORPUS = {
    "type": "function",
    "function": {
        "name": "grep_corpus",
        "description": "Exact regex/pattern matching on the corpus. Use for specific names, dates, numbers, codes, or exact phrases that semantic search might miss.",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regex pattern or exact string to search for.",
                }
            },
            "required": ["pattern"],
        },
    },
}

READ_DOCUMENT = {
    "type": "function",
    "function": {
        "name": "read_document",
        "description": "Read a document's full text content. Use whenever a snippet looks promising but incomplete — full text reveals connections snippets hide.",
        "parameters": {
            "type": "object",
            "properties": {
                "doc_id": {
                    "type": "string",
                    "description": "Document ID to read in full.",
                }
            },
            "required": ["doc_id"],
        },
    },
}

CURATE = {
    "type": "function",
    "function": {
        "name": "curate",
        "description": "Update your curated evidence set. Add promising documents and remove irrelevant ones. Max 30 curated documents. Curate after EVERY search — do not skip curation.",
        "parameters": {
            "type": "object",
            "properties": {
                "add_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Document IDs to add to curated set.",
                },
                "remove_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Document IDs to remove from curated set.",
                },
            },
            "required": ["add_ids"],
        },
    },
}

VERIFY = {
    "type": "function",
    "function": {
        "name": "verify",
        "description": "Record whether specific in-memory documents support a claim. Use on the key claims your evidence must carry before calling end_search — verification records travel with the final curated set.",
        "parameters": {
            "type": "object",
            "properties": {
                "doc_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Document IDs you checked (must already be in the candidate pool or curated set).",
                },
                "claim": {
                    "type": "string",
                    "description": "The claim being checked against these documents.",
                },
                "supports": {
                    "type": "boolean",
                    "description": "true if the documents support the claim; false if they contradict or fail to support it.",
                },
                "reasoning": {
                    "type": "string",
                    "description": "Brief justification citing the documents.",
                },
            },
            "required": ["doc_ids", "claim", "supports"],
        },
    },
}

END_SEARCH = {
    "type": "function",
    "function": {
        "name": "end_search",
        "description": "Submit your final curated evidence set and conclude the search. Only call when you have thoroughly covered the query and gathered sufficient evidence.",
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "Brief explanation of why you have sufficient evidence and what your curated set covers.",
                }
            },
            "required": ["reasoning"],
        },
    },
}

ALL_TOOLS = [FAN_OUT_SEARCH, SEARCH_CORPUS, GREP_CORPUS, READ_DOCUMENT, CURATE, VERIFY, END_SEARCH]
