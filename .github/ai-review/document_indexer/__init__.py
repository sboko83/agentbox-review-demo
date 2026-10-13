"""Reusable local document indexer. Imports have no model or file side effects."""
from .config import Config, load_config
from .pipeline import build_index, search_index, SearchIndex
from .types import Document, Chunk, SearchResult


def compare_strategies(*args, **kwargs):
    from .evaluation import compare_strategies as compare
    return compare(*args, **kwargs)


__all__ = ["Config", "load_config", "Document", "Chunk", "SearchResult",
           "build_index", "search_index", "SearchIndex", "compare_strategies"]
