"""Versioned SQLite storage, published only after integrity validation."""
import json
import os
import sqlite3
import tempfile
from dataclasses import asdict
from pathlib import Path
from .types import Chunk
from .embeddings import validate_vectors

SCHEMA_VERSION = 1
NORMALIZATION_VERSION = 1
SUPPORTED_CHUNKING_VERSIONS = (2,)


class SQLiteStore:
    def save(self, output, metadata, documents, chunks, vectors):
        return save_index(output, metadata, documents, chunks, vectors)

    def load(self, path):
        return load_index(path)


def save_index(output, metadata, documents, chunks, vectors):
    vectors = validate_vectors(vectors, len(chunks))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=output.name + ".", suffix=".tmp", dir=output.parent)
    os.close(fd)
    try:
        with sqlite3.connect(temporary) as db:
            db.execute("PRAGMA foreign_keys=ON")
            db.executescript("""
                PRAGMA user_version=1;
                CREATE TABLE builds (id INTEGER PRIMARY KEY, status TEXT NOT NULL, metadata TEXT NOT NULL);
                CREATE TABLE documents (source_id TEXT PRIMARY KEY, metadata TEXT NOT NULL);
                CREATE TABLE chunks (
                    chunk_id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL REFERENCES documents(source_id),
                    metadata TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    dimension INTEGER NOT NULL,
                    dtype TEXT NOT NULL,
                    byte_order TEXT NOT NULL
                );
            """)
            db.execute("INSERT INTO builds VALUES (1, 'complete', ?)", (json.dumps(metadata),))
            db.executemany("INSERT INTO documents VALUES (?, ?)",
                           [(d.source_id, json.dumps(asdict(d))) for d in documents])
            rows = [(c.chunk_id, c.source_id, json.dumps(asdict(c)), v.astype("<f4").tobytes(), len(v))
                    for c, v in zip(chunks, vectors)]
            db.executemany("INSERT INTO chunks VALUES (?, ?, ?, ?, ?, 'float32', 'little')", rows)
            intact = db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
            if not intact or db.execute("PRAGMA foreign_key_check").fetchall():
                raise ValueError("SQLite integrity check failed")
        load_index(temporary)
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_index(path):
    import numpy as np
    path = Path(path).resolve()
    if not path.is_file():
        raise ValueError(f"Index does not exist: {path}")
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as db:
        if db.execute("PRAGMA user_version").fetchone()[0] != SCHEMA_VERSION:
            raise ValueError("Unsupported index schema; rebuild index")
        build = db.execute("SELECT status, metadata FROM builds WHERE id=1").fetchone()
        if not build or build[0] != "complete":
            raise ValueError("Index is incomplete; rebuild index")
        metadata = json.loads(build[1])
        if metadata.get("schema_version") != SCHEMA_VERSION \
                or metadata.get("normalization_version") != NORMALIZATION_VERSION \
                or metadata.get("chunking_version") not in SUPPORTED_CHUNKING_VERSIONS:
            raise ValueError("Unsupported index preprocessing version; rebuild index")
        rows = db.execute("SELECT metadata, embedding, dimension, dtype, byte_order "
                          "FROM chunks ORDER BY chunk_id").fetchall()
    if not rows:
        raise ValueError("Index contains no chunks; rebuild index")
    chunks, vectors = [], []
    for raw, blob, dimension, dtype, order in rows:
        valid = dtype == "float32" and order == "little" and len(blob) == dimension * 4
        if not valid or dimension != metadata["dimension"]:
            raise ValueError("Incompatible vector format; rebuild index")
        chunks.append(Chunk(**json.loads(raw)))
        vectors.append(np.frombuffer(blob, dtype="<f4"))
    return metadata, chunks, validate_vectors(np.stack(vectors), len(chunks), metadata["dimension"])
