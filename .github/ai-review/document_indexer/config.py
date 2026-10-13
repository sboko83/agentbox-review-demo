"""Versioned configuration; paths are relative to their declaring file."""
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path

REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
STRATEGIES = ("fixed", "structure", "sentence", "recursive")
PATH_KEYS = ("root", "output", "cache", "manifest", "questions")


@dataclass(frozen=True)
class Config:
    root: str
    output: str
    cache: str
    version: int = 1
    include: tuple[str, ...] = ("**/*.md", "**/*.txt")
    exclude: tuple[str, ...] = ()
    manifest: str | None = None
    strategy: str = "structure"
    target_tokens: int = 384
    overlap_tokens: int = 48
    model: str = "intfloat/multilingual-e5-small"
    model_revision: str = REVISION
    batch_size: int = 16
    device: str = "cpu"
    language: str = "unknown"
    questions: str | None = None
    embedding_cache: bool = True

    def __post_init__(self):
        if self.version != 1:
            raise ValueError("Unsupported configuration version; migrate to version 1")
        if self.strategy not in STRATEGIES:
            raise ValueError("strategy must be one of: " + ", ".join(STRATEGIES))
        for name in ("target_tokens", "overlap_tokens", "batch_size"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        if not 0 <= self.overlap_tokens < self.target_tokens <= 480:
            raise ValueError("Require 0 <= overlap_tokens < target_tokens <= 480")
        if type(self.embedding_cache) is not bool:
            raise ValueError("embedding_cache must be a boolean")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not re.fullmatch(r"[a-f0-9]{40}", self.model_revision):
            raise ValueError("model_revision must be a pinned 40-character commit")
        for name in ("include", "exclude"):
            values = getattr(self, name)
            if not isinstance(values, (list, tuple)) or not all(isinstance(v, str) for v in values):
                raise ValueError(f"{name} must be a list of patterns")
            if any(Path(v).is_absolute() or ".." in Path(v).parts for v in values):
                raise ValueError("Source patterns must stay within root")

    def to_dict(self):
        return asdict(self)


def load_config(path, **overrides):
    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    defaults = {"root": ".", "output": "artifacts/index.sqlite", "cache": ".cache"}
    defaults.update(data)
    for key in PATH_KEYS:
        if defaults.get(key) is not None:
            defaults[key] = str((path.parent / defaults[key]).resolve())
    for key, value in overrides.items():
        if value is not None:
            defaults[key] = str(Path(value).resolve()) if key in PATH_KEYS else value
    return Config(**defaults)
