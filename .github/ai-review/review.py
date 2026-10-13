#!/usr/bin/env python3
"""AI review of a pull request: diff → RAG over docs and code → LLM → PR comment.

Each step is a subcommand, so each shows up as its own step of the Actions run;
state passes between them through --work (default .ai-review):
    review.py diff   --base SHA --head SHA     changed files, numbered hunks  → diff.json
    review.py index  --base SHA                docs + code of the base commit → index.sqlite
    review.py review --model ID                retrieval + model call         → review.json, comment.md
    review.py post   --github-repo O/N --pr N  one PR comment, updated in place on every push
    review.py all    --base SHA --head SHA --model ID   the first three, for local runs
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Locally the indexer and the day-31 schema renderer live in sibling tasks;
# in the demo repository they are copied next to this file.
for extra in (HERE, HERE.parents[1] / "week-5-task-1", HERE.parents[1] / "week-7-task-1" / "tools"):
    if extra.is_dir():
        sys.path.insert(0, str(extra))

import corpus  # noqa: E402
import zen  # noqa: E402

MARKER = "<!-- ai-review -->"
DIFF_LIMIT = 40_000  # characters of numbered diff sent to the model
FILE_LIMIT = 12_000
QUERY_CHARS = 700  # e5 takes 512 tokens; code is ~1.5 chars per token at worst
TOP_DOCS, TOP_SCHEMA, TOP_CODE = 4, 1, 4
SKIP = re.compile(r"(^|/)(package-lock\.json|.*\.lock|.*\.min\.js|.*\.svg|.*\.png)$")
HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@ ?(.*)$")
STATUS = {"A": "новый", "M": "изменён", "D": "удалён", "R": "переименован"}
SECTIONS = (
    ("bugs", "🐞 Потенциальные баги"),
    ("architecture", "🏛 Архитектурные проблемы"),
    ("recommendations", "💡 Рекомендации"),
)

SYSTEM = """Ты — ревьюер кода проекта. Тебе дают diff пулл-реквеста и фрагменты проекта
[S1], [S2], …, найденные поиском по документации и коду базовой ветки.

Найди в изменениях:
- bugs — потенциальные баги: ошибки логики, необработанные случаи, падения,
  неверные коды ответа, потеря данных, гонки;
- architecture — архитектурные проблемы: нарушение решений и инвариантов из
  документации, обход слоёв и портов, дублирование уже существующего кода;
- recommendations — рекомендации: тесты, валидация, читаемость. Новое поведение
  без теста в этом же diff — это рекомендация.

Правила:
- Ревьюй только добавленные строки (с «+» в diff), существующий код не трогай.
- Баг — только то, что ломается при конкретном вводе или состоянии; назови его в
  details. Догадку без такого сценария не пиши.
- Не советуй тест, который уже есть в этом diff.
- line — номер строки новой версии файла из левой колонки diff.
- Нарушение решения или инварианта из фрагментов документации — это architecture,
  даже если оно ведёт к багу. Одну проблему не повторяй в двух разделах.
- Архитектурное замечание обязано опираться на фрагмент: перечисли его в sources
  и сошлись в тексте как [S2]. Нет подходящего фрагмента — нет замечания.
- Ссылайся на [Sn], только если фрагмент действительно подтверждает сказанное.
- Не выдумывай. Нечего сказать в разделе — пустой список.
- По-русски и коротко: title — одна строка; details — 1–2 предложения, что не так
  и чем грозит; fix — одно предложение, как исправить.

Ответ — только JSON, без пояснений и без ```:
{"summary": "одна фраза об изменении и главном риске",
 "bugs": [{"file": "путь", "line": 12, "title": "", "details": "", "fix": "", "sources": ["S1"]}],
 "architecture": [...], "recommendations": [...]}"""


PLAN = """Ты готовишь поиск по документации проекта перед ревью пулл-реквеста.
По diff и описанию сформулируй 2–4 поисковых запроса на русском: какие решения,
инварианты и слои проекта затрагивает изменение. Запрос — короткая фраза про суть
(«неизменяемые версии профиля», «доступ к хранилищу из API»), без имён функций и кода.

Ответ — только JSON: {"queries": ["…", "…"]}"""


def say(text: str) -> None:
    print(text, flush=True)


def git(*args: str, cwd: Path | None = None) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True,
                          text=True, encoding="utf-8", errors="replace").stdout


# ---------------------------------------------------------------- diff

def parse_diff(text: str) -> list[dict]:
    """Unified diff → [{path, hunks: [{header, lines: [(mark, old_no, new_no, text)]}]}]."""
    files, current, hunk, old_no, new_no = [], None, None, 0, 0
    for line in text.splitlines():
        if line.startswith("diff --git "):
            current = {"path": line.split(" b/", 1)[-1], "hunks": [], "added": 0, "removed": 0}
            files.append(current)
            hunk = None
        elif current is None:
            continue
        elif line.startswith("+++ "):
            if line[4:] != "/dev/null":
                current["path"] = line[6:] if line.startswith("+++ b/") else line[4:]
        elif line.startswith(("--- ", "index ", "new file", "deleted file", "similarity",
                              "rename ", "old mode", "new mode", "Binary files")):
            continue
        elif match := HUNK.match(line):
            old_no, new_no = int(match[1]), int(match[3])
            hunk = {"header": match[5].strip(), "lines": []}
            current["hunks"].append(hunk)
        elif hunk is not None and line[:1] in ("+", "-", " ", ""):
            mark, body = (line[:1] or " "), line[1:]
            if mark == "+":
                hunk["lines"].append(("+", None, new_no, body))
                new_no += 1
                current["added"] += 1
            elif mark == "-":
                hunk["lines"].append(("-", old_no, None, body))
                old_no += 1
                current["removed"] += 1
            else:
                hunk["lines"].append((" ", old_no, new_no, body))
                old_no += 1
                new_no += 1
    return files


def render_file(file: dict) -> str:
    head = f"### {file['path']} ({file['status']}, +{file['added']} −{file['removed']})\n"
    parts = [head]
    for hunk in file["hunks"]:
        parts.append(f"@@ {hunk['header']}\n".replace("@@ \n", "@@\n"))
        for mark, _old, new, body in hunk["lines"]:
            number = f"{new:>5}" if new is not None else "     "
            parts.append(f"{number} {mark} {body}\n")
    text = "".join(parts)
    if len(text) > FILE_LIMIT:
        text = text[:FILE_LIMIT] + f"\n… файл обрезан, показано {FILE_LIMIT} символов\n"
    return text


def collect_diff(base: str, head: str, repo: Path) -> dict:
    statuses = {}
    for row in git("diff", "--name-status", "--find-renames", f"{base}...{head}", cwd=repo).splitlines():
        parts = row.split("\t")
        statuses[parts[-1]] = STATUS.get(parts[0][:1], parts[0])
    raw = git("diff", "--no-color", "--unified=4", "--find-renames", f"{base}...{head}", cwd=repo)
    files = [f for f in parse_diff(raw) if not SKIP.search(f["path"]) and f["hunks"]]
    for file in files:
        file["status"] = statuses.get(file["path"], "изменён")
    text, shown = "", []
    for file in files:
        block = render_file(file)
        if len(text) + len(block) > DIFF_LIMIT:
            text += f"\n… ещё файлов не показано: {len(files) - len(shown)}\n"
            break
        text += block + "\n"
        shown.append(file["path"])
    # Refs such as main or HEAD move; the comment and its links need the commits reviewed.
    base, head = (git("rev-parse", ref, cwd=repo).strip() for ref in (base, head))
    return {"base": base, "head": head, "files": files, "text": text,
            "added": sum(f["added"] for f in files), "removed": sum(f["removed"] for f in files)}


def cmd_diff(args) -> int:
    data = collect_diff(args.base, args.head, args.repo)
    save(args.work, "diff.json", data)
    say(f"Изменено файлов: {len(data['files'])}, +{data['added']} −{data['removed']}")
    for file in data["files"]:
        say(f"  {file['status']:>12}  {file['path']}  +{file['added']} −{file['removed']}")
    return 0


# ---------------------------------------------------------------- index

def indexer_config(args):
    from document_indexer.config import load_config
    return load_config(HERE / "index.json", root=str(args.work / "corpus"),
                       output=str(args.work / "index.sqlite"), cache=str(args.cache))


def cmd_index(args) -> int:
    from document_indexer.pipeline import build_index

    base_tree = args.work / "base"
    if base_tree.exists():
        shutil.rmtree(base_tree)
    base_tree.mkdir(parents=True)
    archive = subprocess.run(["git", "archive", args.base], cwd=args.repo, check=True,
                             capture_output=True).stdout
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(base_tree, filter="data")
    counts = corpus.build(base_tree, args.work / "corpus")
    say(f"Корпус базовой ветки {args.base[:7]}: документация {counts['doc']}, "
        f"схема {counts['schema']}, код {counts['code']} файлов")
    meta = build_index(indexer_config(args))
    say(f"Индекс: {meta['chunks']} фрагментов, эмбеддингов посчитано {meta['embeddings_computed']}, "
        f"из кэша {meta['embeddings_cached']}, {meta['embedding_seconds']:.1f} с")
    return 0


# ---------------------------------------------------------------- review

def diff_queries(diff: dict) -> list[str]:
    """One code query per changed file: path, hunk headers and added lines."""
    out = []
    for file in diff["files"]:
        headers = " ".join(dict.fromkeys(h["header"] for h in file["hunks"] if h["header"]))
        added = "\n".join(t for hunk in file["hunks"] for m, _o, _n, t in hunk["lines"]
                          if m == "+" and t.strip())
        out.append(f"{file['path']} {headers}\n{added}"[:QUERY_CHARS])
    return out


def pr_query(title: str, body: str) -> list[str]:
    text = f"{title}\n{body}".strip()
    return [text[:QUERY_CHARS]] if text else []


def parse_queries(text: str) -> list[str]:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("в ответе нет JSON-объекта")
    found = json.loads(text[start:end + 1]).get("queries") or []
    return [str(q).strip()[:QUERY_CHARS] for q in found if str(q).strip()][:4]


def best(results: list, kinds: tuple[str, ...], top: int) -> list[dict]:
    """Best score per chunk of the given kinds across all queries' results."""
    found: dict[str, dict] = {}
    for result in results:
        chunk = result.chunk
        if not chunk.file.startswith(tuple(k + "/" for k in kinds)):
            continue
        if chunk.chunk_id not in found or result.score > found[chunk.chunk_id]["score"]:
            found[chunk.chunk_id] = {"chunk": chunk, "score": result.score, **corpus.source_of(chunk)}
    return sorted(found.values(), key=lambda r: -r["score"])[:top]


def select(doc_results: list, code_results: list, changed: set[str]) -> list[dict]:
    """Documentation and schema (searched by meaning), then code (searched by the diff).

    Each kind has its own slots: SQL resembles any query about data and would crowd out prose.
    """
    picked = (best(doc_results, ("doc",), TOP_DOCS) + best(doc_results, ("schema",), TOP_SCHEMA)
              + best(code_results, ("code",), TOP_CODE))
    for n, item in enumerate(picked, 1):
        item["label"] = f"S{n}"
        item["changed"] = item["path"] in changed
    return picked


def where(item: dict) -> str:
    lines = f":{item['start']}–{item['end']}" if item["start"] else ""
    section = f" › {item['section']}" if item["section"] else ""
    return f"{item['path']}{lines}{section}"


def build_messages(diff: dict, context: list[dict], title: str = "", body: str = "") -> list[dict]:
    fragments = "\n\n".join(f"[{c['label']}] {where(c)}\n{c['chunk'].text.strip()}" for c in context)
    pr = f"Пулл-реквест: {title}\n{body}".strip() if title or body else "Пулл-реквест без описания."
    user = (f"{pr}\n\n## Фрагменты проекта\n\n{fragments}\n\n"
            f"## Diff (номер строки новой версии · знак · код)\n\n{diff['text']}")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def parse_review(text: str) -> dict:
    """Model text → review dict; tolerates code fences and prose around the JSON object."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("в ответе нет JSON-объекта")
    data = json.loads(text[start:end + 1])
    review = {"summary": str(data.get("summary", "")).strip()}
    for key, _title in SECTIONS:
        items = data.get(key) or []
        review[key] = [
            {"file": str(i.get("file", "")), "line": i.get("line") if isinstance(i.get("line"), int) else None,
             "title": str(i.get("title", "")).strip(), "details": str(i.get("details", "")).strip(),
             "fix": str(i.get("fix", "")).strip(),
             "sources": _sources(i)}
            for i in items if isinstance(i, dict) and i.get("title")
        ]
    return review


def _sources(item: dict) -> list[str]:
    """Declared sources plus [Sn] cited in the text: models sometimes cite only there."""
    declared = [str(s).strip("[] ") for s in item.get("sources") or []]
    cited = re.findall(r"\[(S\d+)\]", f"{item.get('details', '')} {item.get('fix', '')}")
    return list(dict.fromkeys(declared + cited))


def blob(repo: str, sha: str, path: str, start=None, end=None) -> str | None:
    if not repo:
        return None
    anchor = f"#L{start}" + (f"-L{end}" if end and end != start else "") if start else ""
    return f"https://github.com/{repo}/blob/{sha}/{path}{anchor}"


def render_comment(review: dict, context: list[dict], diff: dict, meta: dict) -> str:
    repo, head, base = meta.get("repo", ""), diff["head"], diff["base"]
    out = [MARKER, "## 🤖 AI-ревью", ""]
    if review["summary"]:
        out += [review["summary"], ""]
    for key, title in SECTIONS:
        items = review[key]
        out.append(f"### {title} ({len(items)})")
        if not items:
            out.append("Не найдено.")
        for item in items:
            place = f"{item['file']}:{item['line']}" if item["line"] else item["file"]
            link = blob(repo, head, item["file"], item["line"])
            place = f"[`{place}`]({link})" if link and item["file"] else f"`{place}`"
            sources = " ".join(f"[{s}]" for s in item["sources"]
                               if f"[{s}]" not in item["details"])
            out.append(f"- **{item['title']}** — {place}  ")
            out.append(f"  {item['details']} {sources}".rstrip() + "  ")
            if item["fix"]:
                out.append(f"  _Исправление:_ {item['fix']}")
        out.append("")
    cited = {s for key, _ in SECTIONS for item in review[key] for s in item["sources"]}
    cited |= set(re.findall(r"\[(S\d+)\]", json.dumps(review, ensure_ascii=False)))
    out += ["<details>",
            f"<summary>Контекст RAG: {len(context)} фрагментов документации и кода базовой ветки, "
            f"использовано в ревью: {len(cited & {c['label'] for c in context})}</summary>", "",
            "Запросы к документации: " + ", ".join(f"«{q}»" for q in meta.get("queries", [])), "",
            "| | Источник | Сходство |", "|---|---|---|"]
    for item in context:
        link = blob(repo, base, item["path"], item["start"], item["end"])
        name = f"[{where(item)}]({link})" if link else f"`{where(item)}`"
        label = f"**{item['label']}**" if item["label"] in cited else item["label"]
        out.append(f"| {label} | {name} | {item['score']:.2f} |")
    out += ["", "</details>", ""]
    cost = f" · ${meta['cost']:.4f}" if meta.get("cost") is not None else ""
    tokens = f"{meta['input_tokens']:,} → {meta['output_tokens']:,}".replace(",", " ")
    out.append(f"<sub>Коммит `{head[:7]}` · файлов: {len(diff['files'])}, +{diff['added']} "
               f"−{diff['removed']} · модель `{meta['model']}` · {tokens} токенов{cost}</sub>")
    return "\n".join(out) + "\n"


def ask_json(model: str, messages: list[dict], api_key: str, parse) -> tuple[object, dict]:
    """Call the model and parse its JSON; one retry with a reminder when parsing fails."""
    answer = zen.complete(model, messages, api_key)
    try:
        return parse(answer["text"]), answer
    except ValueError as error:  # json.JSONDecodeError is a ValueError
        say(f"Ответ не разобран ({error}), повторяю запрос")
        messages = messages + [{"role": "assistant", "content": answer["text"]},
                               {"role": "user", "content": "Это не JSON. Верни только JSON-объект по схеме."}]
        retry = zen.complete(model, messages, api_key)
        retry["input_tokens"] += answer["input_tokens"]
        retry["output_tokens"] += answer["output_tokens"]
        return parse(retry["text"]), retry


def cmd_review(args) -> int:
    from document_indexer.pipeline import SearchIndex

    diff = load(args.work, "diff.json")
    if not diff["files"]:
        (args.work / "comment.md").write_text(
            f"{MARKER}\n## 🤖 AI-ревью\n\nВ PR нет файлов для ревью (только lock-файлы и бинарные).\n",
            encoding="utf-8")
        say("Нечего ревьюить")
        return 0
    title, body = pr_text()
    api_key = os.environ.get("OPENCODE_API_KEY") or read_env_key()
    if not api_key:
        say("Нет OPENCODE_API_KEY")
        return 2

    pr = f"Пулл-реквест: {title}\n{body}".strip() if title or body else "Пулл-реквест без описания."
    plan, planned = ask_json(args.model, [{"role": "system", "content": PLAN},
                                          {"role": "user", "content": f"{pr}\n\n{diff['text']}"}],
                             api_key, parse_queries)
    say("Запросы к документации (сформулировала модель):")
    for query in plan:
        say(f"  «{query}»")

    index = SearchIndex(str(args.work / "index.sqlite"))
    # Every chunk is scored: code outnumbers docs ~15:1 and would push them out of any top-k.
    search = lambda qs: [r for q in qs for r in index.search(q, len(index.chunks))]  # noqa: E731
    context = select(search(plan + pr_query(title, body)), search(diff_queries(diff)),
                     {f["path"] for f in diff["files"]})
    say("Найдено поиском (RAG):")
    for item in context:
        say(f"  [{item['label']}] {item['score']:.2f}  {where(item)}")

    messages = build_messages(diff, context, title, body)
    say(f"Модель {args.model}: отправлено ~{sum(len(m['content']) for m in messages) // 4:,} токенов…"
        .replace(",", " "))
    review, answer = ask_json(args.model, messages, api_key, parse_review)
    answer["input_tokens"] += planned["input_tokens"]
    answer["output_tokens"] += planned["output_tokens"]

    prices = zen.price(args.model)
    meta = {"model": args.model, "queries": plan, "repo": os.environ.get("GITHUB_REPOSITORY", args.github_repo or ""),
            "input_tokens": answer["input_tokens"], "output_tokens": answer["output_tokens"],
            "cost": (answer["input_tokens"] * prices[0] + answer["output_tokens"] * prices[1]) / 1e6
            if prices else None}
    save(args.work, "review.json", {"review": review, "meta": meta, "raw": answer["text"],
                                    "context": [{k: v for k, v in c.items() if k != "chunk"}
                                                for c in context]})
    comment = render_comment(review, context, diff, meta)
    (args.work / "comment.md").write_text(comment, encoding="utf-8")
    say(f"Ревью: баги {len(review['bugs'])}, архитектура {len(review['architecture'])}, "
        f"рекомендации {len(review['recommendations'])} · {meta['input_tokens']} → "
        f"{meta['output_tokens']} токенов")
    for key, title in SECTIONS:
        for item in review[key]:
            say(f"  {title.split(' ', 1)[1]}: {item['title']} ({item['file']}:{item['line']})")
    return 0


# ---------------------------------------------------------------- post

def github(method: str, url: str, token: str, body: dict | None = None, opener=urllib.request.urlopen):
    request = urllib.request.Request(
        url, method=method, data=json.dumps(body).encode() if body is not None else None,
        headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
                 "User-Agent": zen.USER_AGENT, "X-GitHub-Api-Version": "2022-11-28"})
    with opener(request, timeout=30) as response:
        return json.loads(response.read() or b"null")


def post_comment(repo: str, pr: int, text: str, token: str, opener=urllib.request.urlopen) -> str:
    """Update the PR's review comment in place, or create it; return its URL."""
    api = f"https://api.github.com/repos/{repo}"
    page = 1
    while True:
        comments = github("GET", f"{api}/issues/{pr}/comments?per_page=100&page={page}", token,
                          opener=opener)
        for comment in comments:
            if comment.get("body", "").startswith(MARKER):
                github("PATCH", f"{api}/issues/comments/{comment['id']}", token, {"body": text}, opener)
                return comment["html_url"]
        if len(comments) < 100:
            break
        page += 1
    return github("POST", f"{api}/issues/{pr}/comments", token, {"body": text}, opener)["html_url"]


def cmd_post(args) -> int:
    text = (args.work / "comment.md").read_text(encoding="utf-8")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write(text)
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        say("Нет GITHUB_TOKEN")
        return 2
    say(f"Комментарий в PR: {post_comment(args.github_repo, args.pr, text, token)}")
    return 0


def cmd_all(args) -> int:
    for step in (cmd_diff, cmd_index, cmd_review):
        code = step(args)
        if code:
            return code
    say(f"\n{(args.work / 'comment.md').read_text(encoding='utf-8')}")
    return 0


# ---------------------------------------------------------------- plumbing

def pr_text() -> tuple[str, str]:
    """Title and body of the PR from the Actions event, if any."""
    path = os.environ.get("GITHUB_EVENT_PATH")
    if not path or not Path(path).exists():
        return os.environ.get("PR_TITLE", ""), os.environ.get("PR_BODY", "")
    pr = json.loads(Path(path).read_text(encoding="utf-8")).get("pull_request") or {}
    return pr.get("title") or "", pr.get("body") or ""


def read_env_key() -> str:
    for env in (HERE.parents[1] / "secrets/.env",):
        if env.exists():
            for line in env.read_text(encoding="utf-8").splitlines():
                if line.startswith("OPENCODE_API_KEY="):
                    return line.split("=", 1)[1].strip()
    return ""


def save(work: Path, name: str, data) -> None:
    work.mkdir(parents=True, exist_ok=True)
    (work / name).write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def load(work: Path, name: str):
    return json.loads((work / name).read_text(encoding="utf-8"))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("command", choices=("diff", "index", "review", "post", "all"))
    parser.add_argument("--base")
    parser.add_argument("--head", default="HEAD")
    parser.add_argument("--model", default=os.environ.get("AI_REVIEW_MODEL", "gpt-6-luna"))
    parser.add_argument("--pr", type=int)
    parser.add_argument("--github-repo", default=os.environ.get("GITHUB_REPOSITORY"))
    parser.add_argument("--repo", type=Path, default=Path("."), help="git checkout to review")
    parser.add_argument("--work", type=Path, default=Path(".ai-review"))
    parser.add_argument("--cache", type=Path,
                        default=Path(os.environ.get("AI_REVIEW_CACHE", "~/.cache/ai-review")).expanduser())
    args = parser.parse_args(argv)
    args.work = args.work.resolve()
    commands = {"diff": cmd_diff, "index": cmd_index, "review": cmd_review, "post": cmd_post,
                "all": cmd_all}
    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
