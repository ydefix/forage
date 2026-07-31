"""
ChromaDB search backend (EXPERIMENTAL — not covered by the offline CI suite).
Supports hybrid search (semantic + keyword) and grep.
"""

import threading
import time
from typing import List, Optional, Tuple



class ChromaSearchBackend:
    """
    ChromaDB-powered search backend with embedding-based semantic search
    and regex grep support.

    Usage:
        backend = ChromaSearchBackend(collection_name="my_corpus")
        backend.index_documents(docs)  # docs = [(id, text), ...]
        results = backend.search("climate change agriculture", top_k=10)
    """

    def __init__(
        self,
        collection_name: str = "search_corpus",
        persist_directory: Optional[str] = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        use_cloud: bool = False,
        cloud_api_key: Optional[str] = None,
        cloud_database: Optional[str] = None,
    ):
        """
        Args:
            collection_name: ChromaDB collection name
            persist_directory: Local directory for persistent storage.
                None = in-memory only.
            embedding_model: SentenceTransformer model for embeddings.
                Default works well for English text.
            use_cloud: Use Chroma Cloud instead of local.
            cloud_api_key: Chroma Cloud API key (if use_cloud=True).
            cloud_database: Chroma Cloud database name (if use_cloud=True).
        """
        try:
            import chromadb
            from chromadb.utils import embedding_functions
        except ImportError as e:
            raise ImportError(
                "ChromaSearchBackend requires chromadb — install with "
                "`pip install forage-agent[chroma]` (or `pip install chromadb`)."
            ) from e
        self.collection_name = collection_name

        if use_cloud:
            self._client = chromadb.CloudClient(
                api_key=cloud_api_key,
                database=cloud_database,
            )
        elif persist_directory:
            self._client = chromadb.PersistentClient(path=persist_directory)
        else:
            self._client = chromadb.Client()

        self._ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )

        # Get or create collection
        try:
            self._collection = self._client.get_collection(
                name=collection_name,
                embedding_function=self._ef,
            )
        except Exception:
            self._collection = self._client.create_collection(
                name=collection_name,
                embedding_function=self._ef,
                metadata={"hnsw:space": "cosine"},
            )

        self._docs_cache: dict = {}  # doc_id -> full text

    def index_documents(
        self,
        documents: List[Tuple[str, str]],
        batch_size: int = 100,
        metadatas: Optional[List[dict]] = None,
    ):
        """
        Index documents into ChromaDB.

        Args:
            documents: List of (doc_id, text) tuples.
            batch_size: Number of documents per batch insert.
            metadatas: Optional metadata per document.
        """
        ids = []
        texts = []
        for doc_id, text in documents:
            ids.append(doc_id)
            texts.append(text)
            self._docs_cache[doc_id] = text

        if metadatas is None:
            metadatas = [{} for _ in documents]

        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i + batch_size]
            batch_texts = texts[i:i + batch_size]
            batch_meta = metadatas[i:i + batch_size]
            self._collection.add(
                ids=batch_ids,
                documents=batch_texts,
                metadatas=batch_meta,
            )

        print(f"Indexed {len(ids)} documents in collection '{self.collection_name}'")

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, str, float]]:
        """
        Semantic search over the corpus.
        Returns list of (doc_id, full_text, score).
        """
        results = self._collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "distances"],
        )

        output = []
        if results["ids"] and results["ids"][0]:
            for doc_id, text, distance in zip(
                results["ids"][0],
                results["documents"][0],
                results["distances"][0],
            ):
                # ChromaDB returns distance (lower = better for cosine).
                # Convert to similarity score (0-1).
                score = 1.0 - (distance / 2.0)  # cosine distance to similarity
                score = max(0.0, min(1.0, score))
                output.append((doc_id, text or self._docs_cache.get(doc_id, ""), score))

        return output

    def grep(self, pattern: str, top_k: int = 10) -> List[Tuple[str, str, float]]:
        """
        Regex pattern matching over the corpus.
        Uses Python re for matching against cached documents.
        """
        import re

        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error:
            # Fall back to substring matching
            pattern_lower = pattern.lower()
            scored = []
            for doc_id, text in self._docs_cache.items():
                count = text.lower().count(pattern_lower)
                if count > 0:
                    scored.append((doc_id, text, min(count / 10.0, 1.0)))
            scored.sort(key=lambda x: -x[2])
            return scored[:top_k]

        scored = []
        for doc_id, text in self._docs_cache.items():
            matches = regex.findall(text)
            if matches:
                score = min(len(matches) / 10.0, 1.0)
                scored.append((doc_id, text, score))

        scored.sort(key=lambda x: -x[2])
        return scored[:top_k]

    def count(self) -> int:
        """Return number of documents in the collection."""
        return self._collection.count()

    def load_corpus_from_directory(
        self, directory: str, glob_pattern: str = "*.txt"
    ):
        """
        Load all text files from a directory as documents.
        Each file becomes one document.
        """
        import glob as glob_mod
        import os

        docs = []
        for filepath in glob_mod.glob(os.path.join(directory, glob_pattern)):
            doc_id = os.path.splitext(os.path.basename(filepath))[0]
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
            docs.append((doc_id, text))

        self.index_documents(docs)
        return len(docs)

    def load_corpus_from_jsonl(
        self, filepath: str, text_key: str = "text", id_key: str = "id"
    ):
        """
        Load documents from a JSONL file.
        Each line is a JSON object with text and optional id.
        """
        import json

        docs = []
        with open(filepath, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue
                obj = json.loads(line)
                doc_id = obj.get(id_key, f"doc_{i:06d}")
                text = obj.get(text_key, "")
                if text:
                    docs.append((doc_id, text))

        self.index_documents(docs)
        return len(docs)
