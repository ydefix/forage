#!/usr/bin/env python3
"""
BrowseComp+-style evaluation for Qwen3.6 search harness.

Evaluates the search agent on a set of benchmark queries with
known gold evidence documents. Reports recall, precision, F1,
trajectory recall, and final-answer recall.

Usage:
    python eval_benchmark.py
"""

import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Set, Tuple

sys.path.insert(0, str(Path(__file__).parent))

from forage import SearchAgent, InMemorySearchBackend
from training.reward import compute_reward, compute_precision, compute_recall, compute_f1


# ─── Benchmark: queries with gold evidence doc IDs ──────────────────────────

BENCHMARK = [
    {
        "query": "What are the impacts of climate change on agriculture?",
        "gold_evidence": {"doc_001", "doc_007", "doc_011"},
        "gold_answer": {"doc_001", "doc_007"},
        "category": "climate",
    },
    {
        "query": "How effective are AI search agents for information retrieval?",
        "gold_evidence": {"doc_003", "doc_004", "doc_008"},
        "gold_answer": {"doc_003", "doc_004"},
        "category": "ai_search",
    },
    {
        "query": "What are the key features of Qwen3.6 and how does it compare to other LLMs?",
        "gold_evidence": {"doc_005", "doc_009"},
        "gold_answer": {"doc_005"},
        "category": "models",
    },
    {
        "query": "What are the main arguments for and against nuclear energy?",
        "gold_evidence": {"doc_013", "doc_002", "doc_007"},
        "gold_answer": {"doc_013"},
        "category": "energy",
    },
    {
        "query": "How do renewable energy sources compare: solar versus wind?",
        "gold_evidence": {"doc_014", "doc_015"},
        "gold_answer": {"doc_014", "doc_015"},
        "category": "energy",
    },
    {
        "query": "What causes deforestation in the Amazon rainforest?",
        "gold_evidence": {"doc_016"},
        "gold_answer": {"doc_016"},
        "category": "environment",
    },
    {
        "query": "How does CRISPR gene editing work and what are its applications?",
        "gold_evidence": {"doc_017"},
        "gold_answer": {"doc_017"},
        "category": "science",
    },
    {
        "query": "What are microplastics and how do they affect human health?",
        "gold_evidence": {"doc_018"},
        "gold_answer": {"doc_018"},
        "category": "health",
    },
]


def evaluate_benchmark(
    benchmark: List[Dict],
    corpus_docs: List[Tuple[str, str]],
    base_url: str = "http://localhost:9393",
    model: str = "Qwen3.6-27B-OptiQ-4bit",
    max_turns: int = 25,
    verbose: bool = True,
) -> Dict:
    """
    Run the full benchmark evaluation.
    
    Returns:
        Dict with overall metrics and per-query details.
    """
    backend = InMemorySearchBackend(corpus_docs)
    agent = SearchAgent(
        base_url=base_url,
        model=model,
        max_turns=max_turns,
        search_backend=backend,
    )
    
    results = []
    
    for i, item in enumerate(benchmark, 1):
        query = item["query"]
        gold_evidence = item["gold_evidence"]
        gold_answer = item.get("gold_answer", gold_evidence)
        category = item.get("category", "general")
        
        if verbose:
            print(f"\n[{i}/{len(benchmark)}] {query[:80]}...")
        
        start_time = time.time()
        
        try:
            memory = agent.search(query)
        except Exception as e:
            if verbose:
                print(f"  [ERROR] {e}")
            results.append({
                "query": query,
                "category": category,
                "error": str(e),
                "recall": 0.0,
                "precision": 0.0,
                "f1": 0.0,
                "turns": 0,
                "curated_count": 0,
                "searches": 0,
                "reward": 0.0,
            })
            continue
        
        elapsed = time.time() - start_time
        
        curated_ids = list(memory.curated_docs.keys())
        pool_ids = set(memory.candidate_pool.keys())
        
        recall = compute_recall(curated_ids, gold_evidence)
        precision = compute_precision(curated_ids, gold_evidence)
        f1 = compute_f1(precision, recall)
        
        reward = compute_reward(
            curated_doc_ids=curated_ids,
            gold_evidence_ids=gold_evidence,
            gold_answer_ids=gold_answer,
            found_pool_ids=pool_ids,
            turns_used=agent._turn_count,
            max_turns=max_turns,
        )
        
        result = {
            "query": query,
            "category": category,
            "recall": round(recall, 4),
            "precision": round(precision, 4),
            "f1": round(f1, 4),
            "turns": agent._turn_count,
            "curated_count": len(memory.curated_docs),
            "searches": len(memory.search_history),
            "reward": round(reward, 4),
            "elapsed_sec": round(elapsed, 1),
            "curated_ids": curated_ids,
        }
        
        results.append(result)
        
        if verbose:
            status = "PASS" if recall >= 0.5 else "FAIL"
            print(f"  [{status}] R={recall:.2f} P={precision:.2f} F1={f1:.2f} "
                  f"turns={agent._turn_count} curated={len(memory.curated_docs)} "
                  f"({elapsed:.1f}s)")
    
    # ─── Aggregate metrics ──────────────────────────────────────────────────
    valid = [r for r in results if "error" not in r]
    
    if not valid:
        return {"results": results, "error": "No successful evaluations"}
    
    avg_recall = sum(r["recall"] for r in valid) / len(valid)
    avg_precision = sum(r["precision"] for r in valid) / len(valid)
    avg_f1 = sum(r["f1"] for r in valid) / len(valid)
    avg_turns = sum(r["turns"] for r in valid) / len(valid)
    avg_curated = sum(r["curated_count"] for r in valid) / len(valid)
    avg_reward = sum(r["reward"] for r in valid) / len(valid)
    total_elapsed = sum(r["elapsed_sec"] for r in valid)
    
    # Per-category breakdown
    categories = {}
    for r in valid:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"recall": [], "precision": [], "f1": []}
        categories[cat]["recall"].append(r["recall"])
        categories[cat]["precision"].append(r["precision"])
        categories[cat]["f1"].append(r["f1"])
    
    cat_summary = {}
    for cat, metrics in categories.items():
        cat_summary[cat] = {
            "recall": round(sum(metrics["recall"]) / len(metrics["recall"]), 4),
            "precision": round(sum(metrics["precision"]) / len(metrics["precision"]), 4),
            "f1": round(sum(metrics["f1"]) / len(metrics["f1"]), 4),
            "count": len(metrics["recall"]),
        }
    
    return {
        "model": model,
        "num_queries": len(valid),
        "average_recall": round(avg_recall, 4),
        "average_precision": round(avg_precision, 4),
        "average_f1": round(avg_f1, 4),
        "average_turns": round(avg_turns, 1),
        "average_curated_docs": round(avg_curated, 1),
        "average_reward": round(avg_reward, 4),
        "total_elapsed_sec": round(total_elapsed, 1),
        "per_category": cat_summary,
        "results": results,
    }


def main():
    from demo import SAMPLE_DOCS
    
    # Use expanded corpus
    all_docs = list(SAMPLE_DOCS)
    
    print("=" * 60)
    print("Qwen3.6 Search Harness — BrowseComp+-style Evaluation")
    print("=" * 60)
    print(f"Corpus: {len(all_docs)} documents")
    print(f"Benchmark: {len(BENCHMARK)} queries")
    print(f"Model: Qwen3.6-27B-OptiQ-4bit (256K context)")
    print()
    
    report = evaluate_benchmark(
        benchmark=BENCHMARK,
        corpus_docs=all_docs,
        base_url="http://localhost:9393",
    )
    
    print("\n" + "=" * 60)
    print("OVERALL RESULTS")
    print("=" * 60)
    print(f"  Queries evaluated: {report['num_queries']}")
    print(f"  Average Recall:    {report['average_recall']:.3f}")
    print(f"  Average Precision: {report['average_precision']:.3f}")
    print(f"  Average F1:        {report['average_f1']:.3f}")
    print(f"  Average Turns:     {report['average_turns']}")
    print(f"  Average Curated:   {report['average_curated_docs']}")
    print(f"  Average Reward:    {report['average_reward']:.3f}")
    print(f"  Total Time:        {report['total_elapsed_sec']:.1f}s")
    
    if report.get("per_category"):
        print(f"\n  Per Category:")
        for cat, metrics in sorted(report["per_category"].items()):
            print(f"    {cat}: R={metrics['recall']:.3f} "
                  f"P={metrics['precision']:.3f} "
                  f"F1={metrics['f1']:.3f} "
                  f"(n={metrics['count']})")
    
    # Save report
    output_path = Path(__file__).parent / "eval_results.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report saved to {output_path}")


if __name__ == "__main__":
    main()
