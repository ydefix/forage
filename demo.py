#!/usr/bin/env python3
"""
Demo: Qwen3.6 search harness via oMLX.

Requires an oMLX endpoint (oMLX :9393). Defaults to http://localhost:9393,
matching harness/agent.py and eval_benchmark.py; point FORAGE_OMLX_URL at the
Studio's tailnet address (e.g. http://<remote-host>:9393) to run remotely.

Usage:
    python demo.py
    FORAGE_OMLX_URL=http://<remote-host>:9393 python demo.py
"""

import os
import sys
from pathlib import Path

# Add harness-search to path
sys.path.insert(0, str(Path(__file__).parent))

from harness import SearchAgent, InMemorySearchBackend

# ─── Sample documents (simulating a retrieval corpus) ────────────────────────

SAMPLE_DOCS = [
    ("doc_001", "Climate change impacts on agriculture include reduced crop yields, increased water stress, and shifting growing seasons. A 2024 IPCC report found that maize yields could drop 24% by 2050 in tropical regions. Adaptation strategies include drought-resistant crops and improved irrigation."),
    ("doc_002", "The Paris Agreement set a goal of limiting global warming to 1.5°C above pre-industrial levels. Agriculture accounts for 23% of global greenhouse gas emissions. Methane from livestock and nitrous oxide from fertilizers are the main agricultural emissions."),
    ("doc_003", "Search agents are AI systems that retrieve and curate information. Key components include a retrieval backend (vector DB), a language model for reasoning, and tools for searching, reading, and curating. Harness-1 demonstrated RL-trained search agents achieving high recall on BrowseComp+."),
    ("doc_004", "The BrowseComp+ benchmark evaluates AI search agents on complex information-seeking tasks requiring multi-step retrieval, evidence curation, and claim verification. Metrics include recall, trajectory recall, final-answer recall, and precision."),
    ("doc_005", "Qwen3.6 is a large language model by Alibaba with native 256K context, strong tool-calling capabilities, and a thinking/reasoning mode. It outperforms GPT-4o on several benchmarks and is available under Apache 2.0 license. Models range from 0.6B to 235B parameters."),
    ("doc_006", "Reinforcement learning for language models (RLHF, GRPO) enables models to optimize for custom reward functions. In search agents, RL rewards can include retrieval recall, answer quality, and trajectory efficiency. Tinker provides a REINFORCE framework for RL training of LLMs."),
    ("doc_007", "Agriculture contributes 10% of US greenhouse gas emissions. Regenerative farming practices like cover cropping, no-till farming, and crop rotation can sequester carbon in soil. The USDA has allocated $3.1 billion for climate-smart agriculture programs."),
    ("doc_008", "Vector databases like ChromaDB, Pinecone, and Weaviate enable semantic search over document embeddings. They support hybrid search combining dense embeddings with sparse keyword retrieval (BM25). This is essential for search agents to find relevant documents efficiently."),
    ("doc_009", "MLX is Apple's machine learning framework optimized for Apple Silicon. It supports quantized models, training, and inference on M-series chips. oMLX provides an OpenAI-compatible API for serving MLX models locally. Qwen3.6-27B runs at 4-bit on M3 Max with 64GB."),
    ("doc_010", "Document chunking strategies affect retrieval quality. Common approaches include fixed-size chunks (512 tokens), semantic chunking by section boundaries, and recursive splitting. Optimal chunk size depends on the embedding model and retrieval task."),
    ("doc_011", "Climate adaptation in farming: Dutch farmers are experimenting with salt-tolerant potatoes as sea levels rise. In India, farmers use AI-powered weather forecasting for planting decisions. California's drought led to a 20% reduction in almond acreage in 2023."),
    ("doc_012", "Evidence curation is the process of selecting, organizing, and verifying retrieved documents to support a claim or answer a question. Best practices: prioritize primary sources, verify claims across multiple documents, note contradictions, and track evidence chains."),
]


def main():
    # ─── Setup backend ───────────────────────────────────────────────────────────
    backend = InMemorySearchBackend(SAMPLE_DOCS)
    print(f"[SETUP] Indexed {len(SAMPLE_DOCS)} documents")
    
    # ─── Setup agent ────────────────────────────────────────────────────────────
    agent = SearchAgent(
        base_url=os.environ.get("FORAGE_OMLX_URL", "http://localhost:9393"),
        model="Qwen3.6-27B-OptiQ-4bit",
        max_turns=20,
        temperature=0.7,
        search_backend=backend,
    )
    
    # ─── Run search ─────────────────────────────────────────────────────────────
    query = "What are the impacts of climate change on agriculture, and what adaptation strategies exist?"
    
    print(f"\n{'='*60}")
    print(f"[QUERY] {query}")
    print(f"{'='*60}\n")
    
    memory = agent.search(query)
    
    # ─── Display results ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"[RESULTS] Search complete")
    print(f"  Turns used: {agent._turn_count}/{agent.max_turns}")
    print(f"  Curated documents: {len(memory.curated_docs)}")
    print(f"  Candidate pool size: {len(memory.candidate_pool)}")
    print(f"  Searches performed: {len(memory.search_history)}")
    
    if memory.curated_docs:
        print(f"\n  === Curated Evidence ===")
        for i, (doc_id, doc) in enumerate(memory.curated_docs.items(), 1):
            # Truncate for display
            text = doc.text[:150].replace("\n", " ") + "..."
            print(f"  [{i}] {doc_id}: {text}")
    else:
        print(f"\n  [WARN] No documents were curated!")
    
    print(f"\n  === Search History ===")
    for i, sr in enumerate(memory.search_history, 1):
        print(f"  [{i}] '{sr.query}' -> {sr.hit_count} hits")


if __name__ == "__main__":
    main()
