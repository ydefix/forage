"""Backend contract tests — all offline, no chromadb required."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from forage.backends import SearchBackend, InMemorySearchBackend  # noqa: E402


class TestBackendContract(unittest.TestCase):
    def test_memory_backend_satisfies_protocol(self):
        backend = InMemorySearchBackend([("d1", "alpha beta"), ("d2", "gamma")])
        self.assertIsInstance(backend, SearchBackend)

    def test_search_and_grep_shapes(self):
        backend = InMemorySearchBackend([("d1", "alpha beta"), ("d2", "gamma alpha")])
        hits = backend.search("alpha", top_k=5)
        self.assertTrue(hits and all(len(h) == 3 for h in hits))
        greps = backend.grep("gam+a", top_k=5)
        self.assertEqual([g[0] for g in greps], ["d2"])

    def test_chroma_module_imports_without_chromadb(self):
        # The module must import cleanly; only the constructor needs chromadb.
        from forage.backends import chroma
        self.assertTrue(hasattr(chroma, "ChromaSearchBackend"))

    def test_chroma_constructor_error_is_actionable_without_chromadb(self):
        try:
            import chromadb  # noqa: F401
            self.skipTest("chromadb installed — guard not exercised")
        except ImportError:
            pass
        from forage.backends import ChromaSearchBackend
        with self.assertRaises(ImportError) as ctx:
            ChromaSearchBackend()
        self.assertIn("forage-agent[chroma]", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
