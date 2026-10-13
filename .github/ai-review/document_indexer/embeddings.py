"""Lazy, pinned local E5 encoder and exact-text vector cache."""
import json
import os
from pathlib import Path
import tempfile
from .documents import digest

MODEL_TOKEN_LIMIT = 512


def validate_vectors(vectors, count=None, dimension=None):
    import numpy as np
    values = np.asarray(vectors, dtype=np.float32)
    shape_ok = values.ndim == 2 and values.shape[1] > 0
    if not shape_ok or (count is not None and len(values) != count) \
            or (dimension is not None and values.shape[1] != dimension):
        raise ValueError("Embedding dimensions do not match; rebuild index/cache")
    normalized = np.allclose(np.linalg.norm(values, axis=1), 1, atol=1e-4)
    if not np.isfinite(values).all() or not normalized:
        raise ValueError("Embeddings must be finite, L2-normalized float32 vectors")
    return values


class LocalEncoder:
    """sentence-transformers model pinned to a revision; no disk cache of its own."""

    def __init__(self, config):
        from sentence_transformers import SentenceTransformer
        self.identity = {
            "model": config.model,
            "revision": config.model_revision,
            "tokenizer_revision": config.model_revision,
            "pooling": "mean",
            "normalization": "l2",
            "preprocessing": 1,
        }
        self.model = SentenceTransformer(
            config.model,
            revision=config.model_revision,
            device=config.device,
            cache_folder=str(Path(config.cache) / "models"),
            trust_remote_code=False,
        )
        self.model.eval()
        self.model.max_seq_length = MODEL_TOKEN_LIMIT
        self.tokenizer = self.model.tokenizer
        self.dimension = self.model.get_sentence_embedding_dimension()
        if not self.tokenizer.is_fast:
            raise ValueError("A fast tokenizer with source offsets is required")
        self.batch_size = config.batch_size

    def encode(self, texts):
        from .chunking import token_count
        for text in texts:
            if token_count(self.tokenizer, text, True) > MODEL_TOKEN_LIMIT:
                raise ValueError("Embedding input exceeds 512 tokens; truncation is forbidden")
        vectors = self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                    convert_to_numpy=True, show_progress_bar=False)
        return validate_vectors(vectors, len(texts), self.dimension)


class VectorCache:
    """Document vectors keyed by encoder identity + exact input text."""

    def __init__(self, directory, identity):
        self.directory = Path(directory)
        self.prefix = json.dumps(identity, sort_keys=True) + "\n"

    def encode(self, encoder, texts):
        """Return (vectors, computed, cached); only missing texts reach the encoder."""
        import numpy as np
        self.directory.mkdir(parents=True, exist_ok=True)
        keys = [digest(self.prefix + text) for text in texts]
        values, missing = {}, {}
        for key, text in zip(keys, texts):
            path = self.directory / (key + ".npy")
            if path.exists():
                values[key] = validate_vectors(np.load(path, allow_pickle=False).reshape(1, -1))[0]
            else:
                missing.setdefault(key, text)
        if missing:
            encoded = validate_vectors(encoder.encode(list(missing.values())), len(missing))
            for key, vector in zip(missing, encoded):
                values[key] = vector
                self._store(key, vector)
        vectors = validate_vectors(np.stack([values[key] for key in keys]), len(texts))
        return vectors, len(missing), len(texts) - len(missing)

    def _store(self, key, vector):
        import numpy as np
        fd, temporary = tempfile.mkstemp(dir=self.directory, suffix=".npy")
        try:
            with os.fdopen(fd, "wb") as stream:
                np.save(stream, vector, allow_pickle=False)
            os.replace(temporary, self.directory / (key + ".npy"))
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
