# Contributing to Forage

Forage is deliberately small: one loop, seven tools, one externalized memory, and a
two-method backend contract. Most contributions fit one of the extension points below —
if yours doesn't, open an issue first and let's talk.

## Extension points (in order of leverage)

1. **Backends** — wire a new corpus by implementing the two-method
   [`SearchBackend`](forage/backends/base.py) protocol (`search`, `grep` — both return
   `(doc_id, full_text, score)` triples). Ship it as `forage/backends/<name>.py` with a
   constructor-time import guard for any heavy dependency (see
   [`chroma.py`](forage/backends/chroma.py)) and an optional extra in `pyproject.toml`.
2. **Tools** — add a schema in [`forage/tools.py`](forage/tools.py), an executor branch in
   `SearchAgent._execute_tool`, and (if the model should be steered) a rule line in
   `SYSTEM_PROMPT`. Every tool must read/write `WorkingMemory` — no hidden state.
3. **Prompts / policies** — the search→curate rhythm, nudges, and budgets live in
   [`forage/agent.py`](forage/agent.py). Behavioral changes need a scripted-loop test.
4. **Training (Phases 2–4)** — SFT trajectory generation and the recall-weighted reward
   live under `training/` in the design (see [DESIGN.md](DESIGN.md)); this is the most
   open research surface.

## The test law

**Every behavioral change ships with a deterministic offline test.** The suite drives the
real loop with a scripted LLM (see `tests/test_harness_loop.py` — no model, no network,
milliseconds). If your change can't be exercised that way, explain why in the PR.

```bash
pip install -e .
python -m unittest discover -s tests -v   # must be green
```

Live runs (optional): any OpenAI-compatible endpoint serving a tool-calling model —
`FORAGE_OMLX_URL=http://<host>:9393 python demo.py`.

## Workflow

- Conventional commits (`feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `chore:`).
- PRs need a semantic title and a link to a canonical issue (the issue templates create
  them) — CI enforces this via the mission-control culture workflow.
- One logical change per PR. Keep the core dependency-free (`requests` only); anything
  heavier goes behind an optional extra.
- This repo is enrolled in Mission Control: agents report progress via
  `.mission-control/AGENT.md`; humans can ignore that machinery entirely.

## Design canon

[DESIGN.md](DESIGN.md) is the adaptation spec (harness-1 → local Qwen3/Gemma). The two
invariants that don't bend: **all state is externalized** (recoverable from the
trajectory), and **Forage returns evidence, never answers**.

Licensed under [Apache-2.0](LICENSE); contributions are accepted under the same terms.
