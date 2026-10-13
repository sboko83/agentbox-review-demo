"""Strict UTF-8 loading with deterministic manifests."""
import fnmatch
import hashlib
import json
from pathlib import Path
from .types import Document


def digest(value):
    return hashlib.sha256(value if isinstance(value, bytes) else value.encode("utf-8")).hexdigest()


def load_documents(config):
    root = Path(config.root).resolve()
    if not root.is_dir():
        raise ValueError(f"Source directory does not exist: {root}")
    manifest = {}
    if config.manifest:
        raw = json.loads(Path(config.manifest).read_text(encoding="utf-8"))
        for entry in raw["documents"]:
            file = entry.get("text_file", entry.get("file"))
            if file in manifest:
                raise ValueError(f"Duplicate manifest path: {file}")
            manifest[file] = entry
    files = sorted({p for pattern in config.include for p in root.glob(pattern) if p.is_file()})
    documents, seen = [], set()
    for path in files:
        file = path.relative_to(root).as_posix()
        if any(fnmatch.fnmatch(file, pattern) for pattern in config.exclude):
            continue
        resolved = path.resolve()
        if not resolved.is_relative_to(root) or resolved in seen:
            raise ValueError(f"Escaping or duplicate source: {file}")
        seen.add(resolved)
        if path.suffix.lower() not in (".md", ".txt"):
            raise ValueError(f"Unsupported source extension: {file}")
        content = path.read_bytes()
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"Source is not UTF-8: {file}") from exc
        if not text.strip():
            raise ValueError(f"Empty source: {file}")
        sha = digest(content)
        entry = manifest.get(file, {})
        if config.manifest and (not entry or entry.get("sha256_text", entry.get("sha256")) != sha):
            raise ValueError(f"Manifest SHA-256 mismatch: {file}")
        heading = next((line.lstrip("# ") for line in text.splitlines() if line.startswith("# ")), None)
        title = entry.get("title") or heading or path.stem
        documents.append(Document(digest(file), file, text, sha, title, entry.get("source"), config.language))
    if not documents:
        raise ValueError("No matching documents")
    if config.manifest:
        for file in manifest:
            selected = any(Path(file).match(pattern)
                           or (pattern.startswith("**/") and Path(file).match(pattern[3:]))
                           for pattern in config.include)
            excluded = any(fnmatch.fnmatch(file, pattern) for pattern in config.exclude)
            if selected and not excluded and file not in {d.file for d in documents}:
                raise ValueError(f"Manifest source is missing: {file}")
    build_manifest = [{"file": d.file, "sha256": d.content_hash} for d in documents]
    return documents, build_manifest, digest(json.dumps(build_manifest, sort_keys=True))
