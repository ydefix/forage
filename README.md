<div align="center">

# FORAGE

**A local search agent that doesn't answer your question — it finds the evidence.**

*Stateful, tool-using retrieval on your own hardware. Forage decomposes a query, hunts a
corpus from every angle, and hands back a curated, verified evidence set — not a confident
paragraph you have to fact-check.*

Apache-2.0 · adapted from [harness-1](https://github.com/pat-jj/harness-1) (arXiv 2606.02373) · an Init & Co. / Ydefix project

**Site → [forage-liart.vercel.app](https://forage-liart.vercel.app)**

</div>

---

## Why

Most "chat with your docs" stacks retrieve exactly once: embed the question, grab top-k
chunks, staple them into a prompt, answer. When the query has three facets, or the key fact
is a name a vector search glides past, nothing goes back to look harder.

**Forage makes the model do the searching.** A small locally-served LLM (Qwen3.6 / Gemma via
any OpenAI-compatible endpoint, e.g. [oMLX](https://github.com/ml-explore/mlx) on Apple
Silicon) runs a real investigation: it decomposes the query, fans out searches, greps for
exact strings, reads promising documents in full, **verifies key claims**, and continuously
curates a working evidence set — stopping only when the evidence holds.

It never writes the final answer. It returns the curated evidence — ranked, deduplicated,
with verification records — for a reader model (or a human) to answer from, with receipts.

## The loop

```
   your query
       │
       ▼
 ┌───────────────────────────────────────────────────────────┐
 │  ≤ 35 turns · any OpenAI-compatible endpoint               │
 │                                                            │
 │   reason ─▶ tool ─┬─ fan_out_search   5 angles at once     │
 │      ▲            ├─ search_corpus    hybrid semantic      │
 │      │            ├─ grep_corpus      names · dates · #s   │
 │      │            ├─ read_document    full text            │
 │      │            ├─ curate           keep the good        │
 │      │            └─ verify           claims ↔ documents   │
 │      │                                                     │
 │      └──── WorkingMemory summary injected every turn ──────│
 │                                                            │
 │   end_search(reasoning) ─▶ the curated evidence set        │
 └───────────────────────────────────────────────────────────┘
```

**All state is externalized** into a `WorkingMemory` (curated docs, candidate pool, search
history, verification records) — nothing hides in the context window. That's what makes runs
replayable, testable, and ultimately trainable.

## Quickstart

```bash
git clone https://github.com/ydefix/forage.git
cd forage

# offline: the full loop, proven with a scripted deterministic LLM — no model needed
python3 -m unittest discover -s tests -v          # 8/8

# live: point it at any OpenAI-compatible endpoint serving a tool-calling model
python3 demo.py                                    # default http://localhost:9393
FORAGE_OMLX_URL=http://<remote-host>:9393 python3 demo.py
```

Requires Python ≥ 3.9 and `requests`. The demo uses a tiny in-memory corpus; wire your own
by implementing the two-method backend interface (`search`, `grep`) — see `harness/backend.py`.

## Layout

| Path | What |
|---|---|
| `harness/agent.py` | `SearchAgent` — the tool loop, system prompt, curate-nudge |
| `harness/state.py` | `WorkingMemory` / `Document` / `SearchResult` / `VerificationRecord` |
| `harness/tools.py` | function-calling schemas for all seven tools |
| `harness/backend.py` | backend interface + in-memory reference implementation |
| `tests/` | deterministic offline loop tests (scripted LLM — no model needed) |
| `demo.py` / `eval_benchmark.py` | runnable entry points (recall / precision / F1) |
| `DESIGN.md` | the harness-1 → Qwen3 adaptation spec and 4-phase plan |

## Status & roadmap

Forage is a research prototype with an honest ledger:

| Phase | What | Status |
|---|---|---|
| 1 | Search harness — loop, 7 tools, externalized state | ✅ built & proven offline |
| 2 | SFT trajectory data (drive a large model through the harness) | ◻ scaffolded |
| 3 | LoRA SFT + GRPO RL (recall-weighted reward) | ◻ planned |
| 4 | BrowseComp+ evaluation vs the harness-1 baseline | ◻ planned |

## License

[Apache-2.0](LICENSE). Adapts the search-harness design of
[harness-1](https://github.com/pat-jj/harness-1) (Apache-2.0) — see [NOTICE](NOTICE).

---

<div align="center">

*Show the evidence, not the answer.*

</div>
