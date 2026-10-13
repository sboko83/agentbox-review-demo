"""Public build and search functions, independent of command-line state."""
from pathlib import Path
import time
from .config import Config
from .documents import load_documents
from .chunking import CHUNKING_VERSION, make_chunks
from .embeddings import LocalEncoder, VectorCache, validate_vectors
from .storage import SCHEMA_VERSION, NORMALIZATION_VERSION, SQLiteStore
from .types import SearchResult


def build_index(config: Config, encoder=None, document_loader=load_documents, store=None):
    started = time.perf_counter()
    documents, manifest, corpus_hash = document_loader(config)
    model_start = time.perf_counter()
    encoder = encoder or LocalEncoder(config)
    model_seconds = time.perf_counter() - model_start
    chunks = [c for document in documents for c in make_chunks(document, config, encoder.tokenizer)]
    if not chunks:
        raise ValueError("No nonempty chunks")
    texts = [c.embedding_text for c in chunks]
    embedding_start = time.perf_counter()
    if config.embedding_cache:
        cache = VectorCache(Path(config.cache) / "vectors", encoder.identity)
        vectors, computed, cached = cache.encode(encoder, texts)
    else:
        vectors, computed, cached = encoder.encode(texts), len(texts), 0
    vectors = validate_vectors(vectors, len(chunks))
    embedding_seconds = time.perf_counter() - embedding_start
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "normalization_version": NORMALIZATION_VERSION,
        "chunking_version": CHUNKING_VERSION,
        "corpus_hash": corpus_hash,
        "manifest": manifest,
        "config": config.to_dict(),
        "model": encoder.identity,
        "dimension": vectors.shape[1],
        "documents": len(documents),
        "chunks": len(chunks),
        "embeddings_computed": computed,
        "embeddings_cached": cached,
        "model_load_seconds": model_seconds,
        "embedding_seconds": embedding_seconds,
        "processing_seconds": time.perf_counter() - started - model_seconds,
    }
    (store or SQLiteStore()).save(config.output, metadata, documents, chunks, vectors)
    return metadata


class SearchIndex:
    def __init__(self, index, encoder=None, store=None):
        self.metadata, self.chunks, self.matrix = (store or SQLiteStore()).load(index)
        self.encoder = encoder or LocalEncoder(Config(**self.metadata["config"]))
        if self.encoder.identity != self.metadata["model"]:
            raise ValueError("Query model does not match index; rebuild index or use its pinned model")

    def search(self, query, k=5):
        import numpy as np
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must not be empty")
        if type(k) is not int or k < 1:
            raise ValueError("k must be positive")
        # Queries are encoded every time: the vector cache holds document chunks only.
        vector = validate_vectors(self.encoder.encode(["query: " + query]), 1, self.matrix.shape[1])[0]
        scores = self.matrix @ vector
        # Stored chunk order supplies deterministic tie-breaking.
        order = np.argsort(-scores, kind="stable")[:k]
        return [SearchResult(self.chunks[int(i)], float(scores[i])) for i in order]


def search_index(index, query, k=5, encoder=None, store=None):
    return SearchIndex(index, encoder, store).search(query, k)
