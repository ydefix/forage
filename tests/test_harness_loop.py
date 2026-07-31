"""
Offline end-to-end test of the Forage search loop — no model server required.

A scripted "LLM" replaces SearchAgent._call_llm and drives the full tool loop
(fan_out_search → curate → grep_corpus → read_document → verify → end_search)
against the InMemorySearchBackend, asserting the externalized WorkingMemory ends
in the expected state. Deterministic by construction: same script, same run.

Run:  python3 -m unittest discover -s tests -v
"""
import json
import sys
import unittest
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from harness import SearchAgent, InMemorySearchBackend  # noqa: E402

DOCS = [
    ("d1", "The aperture accent is lavender #B4A8FF in the Environ Console world."),
    ("d2", "Bayer-8 ordered dithering maps density to a 1-bit ink field."),
    ("d3", "The training reward is recall-weighted with penalties for wasted turns."),
    ("d4", "Bananas are yellow and entirely unrelated to retrieval."),
]


def tc(n, name, **args):
    """Build one OpenAI-shaped tool call."""
    return {"id": f"call_{n}", "function": {"name": name, "arguments": json.dumps(args)}}


def scripted(agent, script):
    """Replace agent._call_llm with a deterministic script; capture what the LLM saw."""
    queue = deque(script)
    seen = []

    def fake_call_llm(messages):
        seen.append([dict(m) for m in messages if isinstance(m, dict)])
        if not queue:
            return None
        content, calls = queue.popleft()
        return {"content": content, "tool_calls": calls, "finish_reason": "tool_calls"}

    agent._call_llm = fake_call_llm
    return seen


class TestHappyPath(unittest.TestCase):
    def setUp(self):
        self.agent = SearchAgent(search_backend=InMemorySearchBackend(DOCS), max_turns=35)
        self.seen = scripted(self.agent, [
            ("plan facets", [tc(1, "fan_out_search", queries=["aperture accent color", "ordered dithering ink"])]),
            ("curate hits", [tc(2, "curate", add_ids=["d1", "d2"])]),
            ("grep exact", [tc(3, "grep_corpus", pattern="recall-weighted")]),
            ("curate grep hit", [tc(4, "curate", add_ids=["d3"])]),
            ("read full text", [tc(5, "read_document", doc_id="d1")]),
            ("verify key claim", [tc(6, "verify", doc_ids=["d1"],
                                     claim="The aperture accent is lavender",
                                     supports=True, reasoning="d1 states it verbatim")]),
            ("done", [tc(7, "end_search", reasoning="evidence sufficient")]),
        ])
        self.memory = self.agent.search("What accent color does the design system use?")

    def test_curated_set(self):
        self.assertEqual(set(self.memory.curated_docs), {"d1", "d2", "d3"})

    def test_search_history_recorded(self):
        # 2 fan-out queries + 1 grep, each externalized into history
        self.assertEqual(len(self.memory.search_history), 3)
        self.assertTrue(self.memory.search_history[-1].query.startswith("grep:"))

    def test_verification_recorded(self):
        recs = self.memory.verification_records
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0].doc_id, "d1")
        self.assertTrue(recs[0].supports)
        self.assertIn("aperture", recs[0].claim)

    def test_loop_terminated_by_end_search(self):
        # 7 scripted turns consumed, well under max_turns
        self.assertEqual(len(self.seen), 7)

    def test_noise_doc_not_curated(self):
        self.assertNotIn("d4", self.memory.curated_docs)


class TestVerifyEdge(unittest.TestCase):
    def test_unknown_doc_ids_skipped(self):
        agent = SearchAgent(search_backend=InMemorySearchBackend(DOCS))
        out = agent._execute_tool("verify", {
            "doc_ids": ["nope"], "claim": "anything", "supports": True})
        self.assertIn("Unknown doc_ids skipped: nope", out)
        self.assertEqual(len(agent._memory.verification_records), 0)

    def test_verify_tool_is_registered(self):
        from harness.tools import ALL_TOOLS
        names = [t["function"]["name"] for t in ALL_TOOLS]
        self.assertIn("verify", names)
        self.assertEqual(len(names), 7)


class TestNudge(unittest.TestCase):
    def test_nudge_after_three_uncurated_turns(self):
        agent = SearchAgent(search_backend=InMemorySearchBackend(DOCS), max_turns=6)
        seen = scripted(agent, [
            ("s1", [tc(1, "search_corpus", query="aperture")]),
            ("s2", [tc(2, "search_corpus", query="dithering")]),
            ("s3", [tc(3, "search_corpus", query="reward")]),
            ("end", [tc(4, "end_search", reasoning="stop")]),
        ])
        agent.search("nudge check")
        # The 4th LLM call must see the [NUDGE] injected after turn 3 with no curation.
        fourth_turn_msgs = seen[3]
        self.assertTrue(any(
            isinstance(m.get("content"), str) and m["content"].startswith("[NUDGE]")
            for m in fourth_turn_msgs))


if __name__ == "__main__":
    unittest.main()
