"""Portable source and result contracts."""
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Document:
    source_id: str
    file: str
    text: str
    content_hash: str
    title: str
    source: str | None = None
    language: str = "unknown"


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source_id: str
    file: str
    text: str
    embedding_text: str
    context: str
    start: int
    end: int
    start_line: int
    end_line: int
    section_path: list[str]
    section_ids: list[str]
    tokens: int
    overlap_tokens: int
    split_reason: str | None = None
    title: str = ""
    source: str | None = None


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float


class Encoder(Protocol):
    identity: dict
    tokenizer: object

    def encode(self, texts: list[str]): ...


class DocumentLoader(Protocol):
    def __call__(self, config) -> tuple[list[Document], list[dict], str]: ...


class IndexStore(Protocol):
    def save(self, output, metadata, documents, chunks, vectors): ...

    def load(self, path): ...
