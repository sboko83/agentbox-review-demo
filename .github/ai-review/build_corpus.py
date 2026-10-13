#!/usr/bin/env python3
"""Assemble the agentbox documentation corpus for the day-21 indexer.

The indexer reads Markdown only, so the data schema (`schema.sql`) and the HTTP
API (OpenAPI from the FastAPI app) are rendered to Markdown, one heading per
table / endpoint group / model: the structure chunker then keeps each in its
own chunk with a breadcrumb. The corpus is derived, so it is rebuilt, not
committed. Stdlib only; OpenAPI comes from agentbox's own venv.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
PROJECT = HERE.parent / "agentbox"
CORPUS = HERE / "corpus"
DOCS = ("README.md", "CHANGELOG.md", "docs/ARCHITECTURE.md")
SCHEMA = "backend/agentbox/store/schema.sql"
OPENAPI_DUMP = (
    "import json; from agentbox.api.app import create_app; "
    "print(json.dumps(create_app().openapi(), ensure_ascii=False))"
)


def split_statements(sql: str) -> list[tuple[str, str]]:
    """(leading comment, statement) pairs; statements end with `;` at line end."""
    pairs, comment, body = [], [], []
    for line in sql.splitlines():
        stripped = line.strip()
        if not body and stripped.startswith("--"):
            comment.append(stripped[2:].strip())
            continue
        if not body and not stripped:
            if comment and not pairs:
                # The file header belongs to no statement.
                comment = []
            continue
        body.append(line)
        if stripped.endswith(";"):
            pairs.append((" ".join(comment), "\n".join(body)))
            comment, body = [], []
    return pairs


TABLE = re.compile(r"CREATE\s+(?:VIRTUAL\s+)?TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(\w+)", re.I)
ON_TABLE = re.compile(r"\bON\s+(\w+)", re.I)


def render_schema(sql: str, source: str) -> str:
    header = [line[2:].strip() for line in sql.splitlines()[:5] if line.startswith("--")]
    tables: dict[str, list[str]] = {}
    other: list[str] = []
    for comment, statement in split_statements(sql):
        block = (f"{comment}\n\n" if comment else "") + f"```sql\n{statement}\n```"
        table = TABLE.search(statement)
        target = ON_TABLE.search(statement)
        if table:
            tables[table[1]] = [block]
        elif target and target[1] in tables:
            tables[target[1]].append(block)
        else:
            other.append(block)
    out = ["# Схема базы данных agentbox", "",
           f"Источник: `{source}`. SQLite. {' '.join(header)}", "",
           "Таблицы: " + ", ".join(f"`{name}`" for name in tables) + ".", ""]
    for name, blocks in tables.items():
        out += [f"## Таблица {name}", "", *("\n\n".join(blocks).splitlines()), ""]
    if other:
        out += ["## Прочие объекты схемы", "", *("\n\n".join(other).splitlines()), ""]
    return "\n".join(out)


def ref_name(schema: dict | None) -> str:
    if not schema:
        return ""
    if "$ref" in schema:
        return schema["$ref"].rsplit("/", 1)[-1]
    if schema.get("type") == "array":
        return f"list[{ref_name(schema.get('items')) or 'any'}]"
    for key in ("oneOf", "anyOf"):
        if key in schema:
            return " | ".join(ref_name(s) or s.get("type", "any") for s in schema[key])
    return schema.get("type", "")


def first_paragraph(text: str | None) -> str:
    return (text or "").strip().split("\n\n", 1)[0].replace("\n", " ")


def render_openapi(spec: dict) -> str:
    groups: dict[str, list[str]] = {}
    overview: dict[str, list[str]] = {}
    for path, ops in spec["paths"].items():
        group = "/".join(path.split("/")[:3])
        for method, op in ops.items():
            # One list per group answers "which endpoints exist" from a single chunk.
            overview.setdefault(group, []).append(f"- `{method.upper()} {path}` — {op.get('summary', '')}")
            lines = [f"### {method.upper()} {path}", "", op.get("summary", "")]
            description = first_paragraph(op.get("description"))
            if description:
                lines += ["", description]
            params = [f"`{p['name']}` ({p['in']})" for p in op.get("parameters", [])]
            if params:
                lines += ["", "Параметры: " + ", ".join(params) + "."]
            body = (op.get("requestBody") or {}).get("content", {})
            for media, content in body.items():
                lines += ["", f"Тело запроса ({media}): `{ref_name(content.get('schema')) or 'object'}`."]
            for code, response in op.get("responses", {}).items():
                if code.startswith("2"):
                    for media, content in response.get("content", {}).items():
                        name = ref_name(content.get("schema"))
                        lines += ["", f"Ответ {code} ({media})" + (f": `{name}`." if name else ".")]
            groups.setdefault(group, []).append("\n".join(lines))
    out = ["# HTTP API agentbox", "",
           f"OpenAPI {spec.get('openapi', '')}, приложение `{spec['info']['title']}` "
           f"версии {spec['info'].get('version', '')}. "
           f"Эндпоинтов: {sum(len(v) for v in spec['paths'].values())}.", "",
           "Группы: " + ", ".join(f"`{g}`" for g in groups) + ".", ""]
    for group, endpoints in groups.items():
        out += [f"## Эндпоинты {group}", "", *overview[group], "", "\n\n".join(endpoints), ""]
    out += ["## Модели данных API", ""]
    for name, schema in spec.get("components", {}).get("schemas", {}).items():
        required = set(schema.get("required", []))
        fields = []
        for field, prop in schema.get("properties", {}).items():
            kind = ref_name(prop) or "any"
            mark = ", обязательное" if field in required else ""
            note = first_paragraph(prop.get("description"))
            fields.append(f"- `{field}`: {kind}{mark}" + (f" — {note}" if note else ""))
        if "enum" in schema:
            fields.append("Значения: " + ", ".join(f"`{v}`" for v in schema["enum"]))
        description = first_paragraph(schema.get("description"))
        out += [f"### {name}", "", *( [description, ""] if description else []),
                *(fields or ["Без полей."]), ""]
    return "\n".join(out)


def dump_openapi(project: Path) -> dict:
    python = project / "backend/.venv/bin/python"
    if not python.exists():
        sys.exit(f"Нет {python}: поставьте окружение agentbox или передайте --openapi FILE")
    result = subprocess.run([str(python), "-c", OPENAPI_DUMP], cwd=project / "backend",
                            capture_output=True, text=True, check=False)
    if result.returncode:
        sys.exit("Выгрузка OpenAPI не удалась:\n" + result.stderr[-2000:])
    return json.loads(result.stdout)


def build(project: Path, corpus: Path, spec: dict) -> list[Path]:
    if corpus.exists():
        shutil.rmtree(corpus)
    written = []
    for name in DOCS:
        target = corpus / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(project / name, target)
        written.append(target)
    generated = {
        corpus / "schema/database.md": render_schema((project / SCHEMA).read_text("utf-8"), SCHEMA),
        corpus / "api/http.md": render_openapi(spec),
    }
    for target, text in generated.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, "utf-8")
        written.append(target)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description="Собрать корпус документации agentbox.")
    parser.add_argument("--project", type=Path, default=PROJECT)
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--openapi", type=Path, help="готовый openapi.json вместо выгрузки")
    args = parser.parse_args()
    spec = json.loads(args.openapi.read_text("utf-8")) if args.openapi else dump_openapi(args.project)
    for path in build(args.project, args.corpus, spec):
        print(f"{path.relative_to(args.corpus)}  {path.stat().st_size} B")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
