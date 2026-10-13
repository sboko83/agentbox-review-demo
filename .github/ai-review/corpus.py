"""Turn a project checkout into a Markdown corpus for the day-21 indexer.

The indexer reads Markdown only and chunks it by headings, so every source kind is
rendered to Markdown with one heading per unit the reviewer may cite:
    doc/<path>             documentation, copied as is (line numbers stay valid)
    schema/<path>.md       SQL schema, one heading per table (day-31 renderer)
    code/<path>.md         Python, one heading per top-level function / class / method,
                           the heading carries the real line range: "`name` · строки 12–40"
`source_of` maps a corpus chunk back to the real file and lines for links.
"""
from __future__ import annotations

import ast
import re
import shutil
from pathlib import Path

DOCS = ("README.md", "docs/**/*.md")
SCHEMAS = ("backend/**/schema.sql",)
CODE = ("backend/**/*.py",)
BIG_CLASS = 60  # lines; longer classes get one heading per method
LINES = re.compile(r"· строки (\d+)–(\d+)$")


def _files(root: Path, patterns: tuple[str, ...]) -> list[Path]:
    return sorted({p for pattern in patterns for p in root.glob(pattern)
                   if p.is_file() and "__pycache__" not in p.parts})


def _fence(lines: list[str], start: int, end: int) -> str:
    return "```python\n" + "".join(lines[start - 1:end]).rstrip() + "\n```\n"


def _units(tree: ast.Module, lines: list[str]):
    """Yield (title, start, end) per citable unit; adjacent module-level code is grouped."""
    loose: list[int] = []
    for node in tree.body:
        start = min([d.lineno for d in getattr(node, "decorator_list", [])] + [node.lineno])
        end = node.end_lineno
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and loose:
            yield "уровень модуля", loose[0], loose[-1]
            loose = []
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield f"`{node.name}`", start, end
        elif isinstance(node, ast.ClassDef):
            methods = [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            if end - start < BIG_CLASS or not methods:
                yield f"`class {node.name}`", start, end
                continue
            first = min([d.lineno for d in methods[0].decorator_list] + [methods[0].lineno])
            yield f"`class {node.name}`", start, first - 1
            for method in methods:
                m_start = min([d.lineno for d in method.decorator_list] + [method.lineno])
                yield f"`{node.name}.{method.name}`", m_start, method.end_lineno
        elif not (node is tree.body[0] and isinstance(node, ast.Expr)
                  and isinstance(node.value, ast.Constant)):
            loose = [loose[0] if loose else start, end]
    if loose:
        yield "уровень модуля", loose[0], loose[-1]


def render_python(source: str, path: str) -> str:
    lines = source.splitlines(keepends=True)
    out = [f"# {path}\n\n"]
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return out[0] + _fence(lines, 1, len(lines))
    doc = ast.get_docstring(tree)
    if doc:
        out.append(doc.strip().split("\n\n")[0] + "\n\n")
    for title, start, end in _units(tree, lines):
        if end >= start:
            out.append(f"## {title} · строки {start}–{end}\n\n{_fence(lines, start, end)}\n")
    return "".join(out)


def render_schema_file(sql: str, path: str) -> str:
    try:
        from build_corpus import render_schema  # day-31 renderer, copied next to this file in CI
        return render_schema(sql, path)
    except ImportError:
        return f"# {path}\n\n```sql\n{sql.rstrip()}\n```\n"


def build(project: Path, corpus: Path) -> dict[str, int]:
    """Write the corpus; return file counts per kind."""
    if corpus.exists():
        shutil.rmtree(corpus)
    counts = {"doc": 0, "schema": 0, "code": 0}
    for kind, patterns in (("doc", DOCS), ("schema", SCHEMAS), ("code", CODE)):
        for src in _files(project, patterns):
            rel = src.relative_to(project).as_posix()
            text = src.read_text(encoding="utf-8", errors="replace")
            if kind == "doc":
                target, body = corpus / "doc" / rel, text
            elif kind == "schema":
                target, body = corpus / "schema" / (rel + ".md"), render_schema_file(text, rel)
            else:
                target, body = corpus / "code" / (rel + ".md"), render_python(text, rel)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(body, encoding="utf-8")
            counts[kind] += 1
    return counts


def source_of(chunk) -> dict:
    """Real path, line range and section names of a corpus chunk.

    `section_path` holds one breadcrumb per section the chunk covers, "Title / A / B";
    the title is the document (for code, its path) and is dropped.
    """
    kind, _, rel = chunk.file.partition("/")
    if kind != "doc":
        rel = rel.removesuffix(".md")
    names, starts, ends = [], [], []
    for crumb in chunk.section_path:
        parts = crumb.split(" / ")[1:]
        if not parts:
            continue
        match = LINES.search(parts[-1])
        if match:
            starts.append(int(match[1]))
            ends.append(int(match[2]))
        name = " › ".join(LINES.sub("", part).strip() for part in parts)
        if name not in names:
            names.append(name)
    if kind == "doc":
        start, end = chunk.start_line, chunk.end_line
    elif starts:
        start, end = min(starts), max(ends)
    else:
        start = end = None
    return {"kind": kind, "path": rel, "start": start, "end": end, "section": ", ".join(names)}
