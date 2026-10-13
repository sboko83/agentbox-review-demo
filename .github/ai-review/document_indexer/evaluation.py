"""Span-based retrieval evaluation with dev-only parameter selection."""
from dataclasses import replace
import importlib.metadata
import json
from math import comb
from pathlib import Path
import platform
import os
import subprocess
import sys
import statistics
import time
from .chunking import parse_blocks
from .documents import load_documents, digest
from .embeddings import LocalEncoder
from .pipeline import build_index, SearchIndex
from .storage import load_index
from .config import STRATEGIES

METRICS = ("hit", "recall", "mrr", "all_facts", "budget_recall")
CONTEXT_BUDGET = 1500
GRID = ((384, 48), (256, 0), (256, 48), (384, 0))
BASELINE = "fixed-384-48"


def covered(span, chunks):
    """True when the union of chunk ranges covers the whole evidence span."""
    intervals = sorted((max(span["start"], c.start), min(span["end"], c.end)) for c in chunks
                       if c.file == span["file"] and c.start < span["end"] and c.end > span["start"])
    cursor = span["start"]
    for start, end in intervals:
        if start > cursor:
            return False
        cursor = max(cursor, end)
    return cursor >= span["end"]


def metrics(question, results):
    spans = question["evidence"]
    chunks = [r.chunk for r in results]
    first = next((rank for rank, result in enumerate(results, 1)
                  if any(covered(span, [result.chunk]) for span in spans)), None)
    return {
        "hit": float(any(covered(span, chunks) for span in spans)),
        "recall": sum(covered(span, chunks) for span in spans) / len(spans),
        "mrr": 1 / first if first else 0.0,
        "all_facts": float(all(covered(span, chunks) for span in spans)),
    }


def aggregate(rows):
    groups = {"all": rows}
    for field in ("language", "type"):
        for value in sorted({r[field] for r in rows}):
            groups[field + ":" + value] = [r for r in rows if r[field] == value]
    return {name: {"n": len(group), **{m: statistics.mean(r[m] for r in group) for m in METRICS}}
            for name, group in groups.items() if group}


def evaluate(index, questions):
    rows = []
    for question in questions:
        started = time.perf_counter()
        results = index.search(question["query"], len(index.chunks))
        row = {
            "id": question["id"],
            "group": question.get("group", question["id"]),
            "query": question["query"],
            "language": question["language"],
            "type": question["type"],
            "query_seconds": time.perf_counter() - started,
            "top5": [{"score": r.score, "chunk_id": r.chunk.chunk_id, "file": r.chunk.file,
                      "start": r.chunk.start, "end": r.chunk.end,
                      "lines": [r.chunk.start_line, r.chunk.end_line]} for r in results[:5]],
        }
        if question["evidence"]:
            # Longest ranked prefix whose full embedding inputs fit the context budget.
            selected, total = [], 0
            for result in results:
                if total + result.chunk.tokens > CONTEXT_BUDGET:
                    break
                selected.append(result)
                total += result.chunk.tokens
            row.update(metrics(question, results[:5]))
            row.update(budget_recall=metrics(question, selected)["recall"], budget_tokens=total)
        rows.append(row)
    return {"metrics": aggregate([r for r in rows if "recall" in r]), "questions": rows}


def paired_topics(rows, baseline_rows):
    """Topic-level Recall@5 wins/losses against a baseline, with an exact sign test.

    RU/EN questions of one topic are averaged first: they are not independent.
    """
    def by_topic(items):
        topics = {}
        for row in items:
            topics.setdefault(row["group"], []).append(row["recall"])
        return {topic: statistics.mean(values) for topic, values in topics.items()}

    current, base = by_topic(rows), by_topic(baseline_rows)
    wins = sum(current[t] > base[t] for t in base)
    losses = sum(current[t] < base[t] for t in base)
    n = wins + losses
    p = min(1.0, 2 * sum(comb(n, i) for i in range(min(wins, losses) + 1)) / 2 ** n) if n else 1.0
    return {"topics": len(base), "wins": wins, "losses": losses, "ties": len(base) - n, "sign_test_p": p}


def structure_stats(documents, chunks):
    """Coverage, duplication and how often chunk ends cut code, tables or sentences."""
    import numpy as np
    total, covered_count, duplicates = 0, 0, 0
    code_breaks, table_breaks, sentence_breaks = 0, 0, 0
    for document in documents:
        counts = np.zeros(len(document.text), dtype=np.int32)
        relevant = [c for c in chunks if c.source_id == document.source_id]
        for chunk in relevant:
            counts[chunk.start:chunk.end] += 1
        meaningful = np.array([not char.isspace() for char in document.text])
        total += int(meaningful.sum())
        covered_count += int(((counts > 0) & meaningful).sum())
        duplicates += int((np.maximum(counts - 1, 0) * meaningful).sum())
        blocks = parse_blocks(document)
        for boundary in {c.end for c in relevant if c.end < len(document.text)}:
            for block in blocks:
                if block.start < boundary < block.end:
                    code_breaks += block.kind == "code"
                    table_breaks += block.kind == "table"
                    unfinished = document.text[:boundary].rstrip()[-1:] not in (".", "!", "?")
                    sentence_breaks += block.kind == "paragraph" and unfinished
    tokens = [c.tokens for c in chunks]
    percentiles = map(float, np.percentile(tokens, [0, 50, 95, 100]))
    return {
        "chunks": len(chunks),
        "tokens": dict(zip(("min", "median", "p95", "max"), percentiles)),
        "small_chunks_under_100_tokens": sum(t < 100 for t in tokens),
        "meaningful_character_coverage": covered_count / total,
        "duplicated_character_ratio": duplicates / total,
        "code_breaks": code_breaks,
        "table_breaks": table_breaks,
        "sentence_breaks_heuristic": sentence_breaks,
        "actual_overlap_tokens": sum(c.overlap_tokens for c in chunks),
    }


def load_questions(path, documents):
    raw = Path(path).read_text(encoding="utf-8")
    dataset = json.loads(raw)
    if dataset["version"] != 1:
        raise ValueError("Unsupported question schema")
    sources = {d.file: d for d in documents}
    for q in dataset["questions"]:
        for span in q["evidence"]:
            document = sources[span["file"]]
            unchanged = span["sha256"] == document.content_hash
            if not unchanged or document.text[span["start"]:span["end"]] != span["quote"]:
                raise ValueError(f"Evidence source changed: {q['id']}")
    splits = {name: [q for q in dataset["questions"] if q["split"] == name]
              for name in ("dev", "test", "no_answer")}
    if not splits["dev"] or not splits["test"]:
        raise ValueError("Both dev and test splits are required")
    groups = [{q.get("group", q["id"]) for q in splits[name]} for name in ("dev", "test")]
    if groups[0] & groups[1]:
        raise ValueError("Question group leaks across dev/test splits")
    return raw, splits


def open_or_build(run_config, encoder, corpus_hash, reuse):
    if not reuse:
        return build_index(run_config, encoder)
    metadata, _, _ = load_index(run_config.output)
    expected = run_config.to_dict()
    fields = ("strategy", "target_tokens", "overlap_tokens", "model", "model_revision")
    if metadata["corpus_hash"] != corpus_hash or metadata["model"] != encoder.identity \
            or any(metadata["config"][field] != expected[field] for field in fields):
        raise ValueError(f"Saved index does not match experiment: {run_config.output}; rebuild it")
    return metadata


def compare_strategies(config, questions=None, output=None, encoder=None, reuse=False, progress=None):
    questions_path = Path(questions or config.questions or "")
    if not questions_path.is_file():
        raise ValueError("A questions JSON file is required")
    documents, _, corpus_hash = load_documents(config)
    raw_questions, splits = load_questions(questions_path, documents)
    directory = Path(output or config.output).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    encoder = encoder or LocalEncoder(config)
    load_seconds = time.perf_counter() - started
    packages = ("numpy", "sentence-transformers", "torch", "transformers", "tokenizers")
    report = {
        "version": 2,
        "corpus_hash": corpus_hash,
        "documents": len(documents),
        "questions_hash": digest(raw_questions),
        "question_counts": {name: len(items) for name, items in splits.items()},
        "model": encoder.identity,
        "device": config.device,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "versions": {name: importlib.metadata.version(name) for name in packages},
        "model_load_seconds": load_seconds,
        "baseline": BASELINE,
        "selection": "Max dev evidence Recall@5, then MRR@5, then fewer chunks; "
                     "test evaluated after selection",
        "runs": {},
        "selected": {},
        "limitations": [
            "RU/EN questions of one topic are dependent; significance is computed per topic",
            "No-answer examples do not calibrate a refusal threshold",
            "Span coverage is a retrieval proxy, not an LLM answer-quality score",
            "Query times use a loaded model; model load is recorded separately",
        ],
    }
    indexes = {}
    for strategy in STRATEGIES:
        for target, overlap in GRID:
            key = f"{strategy}-{target}-{overlap}"
            if progress:
                progress(f"{'Open' if reuse else 'Build'} {key}; evaluate dev")
            path = str(directory / (key + ".sqlite"))
            run_config = replace(config, strategy=strategy, target_tokens=target,
                                 overlap_tokens=overlap, output=path)
            started = time.perf_counter()
            metadata = open_or_build(run_config, encoder, corpus_hash, reuse)
            seconds = time.perf_counter() - started
            index = indexes[key] = SearchIndex(path, encoder)
            report["runs"][key] = {
                "index": path,
                "reused": reuse,
                "build_seconds": None if reuse else seconds,
                "index_open_seconds": seconds if reuse else None,
                "embedding_seconds": metadata["embedding_seconds"],
                "embeddings_computed": metadata.get("embeddings_computed"),
                "index_bytes": Path(path).stat().st_size,
                "structure": structure_stats(documents, index.chunks),
                "dev": evaluate(index, splits["dev"]),
            }

        def dev_rank(key):
            dev = report["runs"][key]["dev"]["metrics"]["all"]
            return dev["recall"], dev["mrr"], -len(indexes[key].chunks)

        report["selected"][strategy] = max((k for k in report["runs"] if k.startswith(strategy + "-")),
                                           key=dev_rank)
    # Selection is fixed before any held-out queries are evaluated.
    held_out = sorted(set(report["selected"].values()) | {f"{s}-384-48" for s in STRATEGIES})
    for key in held_out:
        if progress:
            progress(f"Evaluate held-out test + no-answer: {key}")
        run = report["runs"][key]
        run["test"] = evaluate(indexes[key], splits["test"])
        run["no_answer"] = evaluate(indexes[key], splits["no_answer"])
        run["cold_process_search_seconds"] = cold_search(run["index"], splits["test"][0]["query"])
    baseline = report["runs"][BASELINE]["test"]["questions"]
    for key in held_out:
        run = report["runs"][key]
        run["test"]["vs_baseline"] = paired_topics(run["test"]["questions"], baseline)
    (directory / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                               encoding="utf-8")
    (directory / "comparison.md").write_text(summary_markdown(report), encoding="utf-8")
    return report


def cold_search(index, query):
    """New offline process: interpreter start, model load from disk and one search."""
    started = time.perf_counter()
    command = ("import json,sys; from document_indexer import search_index; "
               "print(json.dumps([r.chunk.chunk_id for r in search_index(sys.argv[1],sys.argv[2])]))")
    environment = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    completed = subprocess.run([sys.executable, "-c", command, index, query], capture_output=True,
                               text=True, env=environment, check=True)
    json.loads(completed.stdout)
    return time.perf_counter() - started


def held_out_runs(report):
    return [(key, run) for key, run in report["runs"].items() if "test" in run]


def retrieval_rows(report):
    """Rows shared by comparison.md and the CLI summary."""
    rows = []
    for key, run in held_out_runs(report):
        m, paired = run["test"]["metrics"]["all"], run["test"]["vs_baseline"]
        versus = "—" if key == report["baseline"] else \
            f"{paired['wins']}/{paired['losses']}/{paired['ties']} (p={paired['sign_test_p']:.2f})"
        rows.append([key, run["structure"]["chunks"], f"{m['hit']:.3f}", f"{m['recall']:.3f}",
                     f"{m['mrr']:.3f}", f"{m['budget_recall']:.3f}", versus])
    return rows


def structure_rows(report):
    rows = []
    for key, run in held_out_runs(report):
        s = run["structure"]
        rows.append([key, int(s["tokens"]["min"]), int(s["tokens"]["median"]), int(s["tokens"]["max"]),
                     s["small_chunks_under_100_tokens"], s["code_breaks"], s["table_breaks"],
                     s["sentence_breaks_heuristic"], f"{s['duplicated_character_ratio']:.1%}"])
    return rows


def findings(report):
    """Factual statements computed from the report; interpretation lives in README."""
    runs = report["runs"]
    base = runs[report["baseline"]]
    best_key, best = max(held_out_runs(report), key=lambda item: (
        item[1]["test"]["metrics"]["all"]["recall"], item[1]["test"]["metrics"]["all"]["mrr"]))
    lines = [f"Лучший Recall@5 на test: {best_key} — {best['test']['metrics']['all']['recall']:.3f}."]
    if best_key != report["baseline"]:
        paired = best["test"]["vs_baseline"]
        lines[0] += (f" Против {report['baseline']} по темам: лучше {paired['wins']}, хуже {paired['losses']}, "
                     f"одинаково {paired['ties']}; sign test p={paired['sign_test_p']:.2f}.")
    significant = [key for key, run in held_out_runs(report)
                   if key != report["baseline"] and run["test"]["vs_baseline"]["sign_test_p"] < 0.05]
    lines.append("Отличие от baseline значимо (p<0.05): " + (", ".join(significant) or "ни у одного варианта") + ".")
    structure = runs["structure-384-48"]["structure"]
    fixed = base["structure"]
    lines.append(
        f"Разрывы code-блоков / предложений: structure-384-48 — {structure['code_breaks']} / "
        f"{structure['sentence_breaks_heuristic']}, {report['baseline']} — "
        f"{fixed['code_breaks']} / {fixed['sentence_breaks_heuristic']}.")
    answered = [q["top5"][0]["score"] for q in base["test"]["questions"]]
    rejected = [q["top5"][0]["score"] for q in base["no_answer"]["questions"]]
    lines.append(
        f"Вопросы без ответа: top-1 cosine {min(rejected):.3f}–{max(rejected):.3f}, у вопросов с ответом "
        f"медиана {statistics.median(answered):.3f}. Порог отказа по cosine здесь не отделяет одно от другого."
        if max(rejected) >= min(answered) else
        f"Вопросы без ответа: top-1 cosine не выше {max(rejected):.3f}, ниже всех вопросов с ответом.")
    return lines


def table(header, rows, align):
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(align) + "|"]
    return lines + ["| " + " | ".join(map(str, row)) + " |" for row in rows]


def examples(report, per_strategy=2):
    """Largest topic-level disagreements with the baseline, one question per topic."""
    base = {q["id"]: q for q in report["runs"][report["baseline"]]["test"]["questions"]}
    lines = []
    for strategy in STRATEGIES:
        key = f"{strategy}-384-48"
        if key == report["baseline"]:
            continue
        differing = [q for q in report["runs"][key]["test"]["questions"]
                     if q["recall"] != base[q["id"]]["recall"]]
        differing.sort(key=lambda q: (-abs(q["recall"] - base[q["id"]]["recall"]), q["id"]))
        seen = set()
        for q in differing:
            if q["group"] in seen or len(seen) == per_strategy:
                continue
            seen.add(q["group"])
            top = q["top5"][0]
            lines.append(f"- {q['query']} Recall@5: fixed={base[q['id']]['recall']:.2f}, "
                         f"{strategy}={q['recall']:.2f}; первый {strategy}: "
                         f"`{top['file']}:{top['lines'][0]}`, cosine={top['score']:.3f}.")
    return lines


def summary_markdown(report):
    counts = report["question_counts"]
    dev_topics = counts["dev"] // 2
    test_topics = counts["test"] // 2
    lines = [
        "# Сравнение чанкинга", "",
        f"Автоматический отчёт. Документов в корпусе: {report['documents']}. Вопросы: "
        f"{dev_topics + test_topics} тем × RU/EN — dev {dev_topics}, test {test_topics} — "
        f"и {counts['no_answer']} без ответа. Параметры выбраны только по dev, метрики ниже — test.", "",
        "## Поиск на test", "",
        *table(["Вариант", "Чанков", "Hit@5", "Recall@5", "MRR@5", f"Recall / {CONTEXT_BUDGET} ток.",
                f"Темы лучше/хуже/равно {report['baseline']}"],
               retrieval_rows(report), ["---", "---:", "---:", "---:", "---:", "---:", "---:"]),
        "", "Выбор на dev: " + ", ".join(report["selected"].values()), "",
        "## Структура чанков", "",
        *table(["Вариант", "Мин. токенов", "Медиана", "Макс.", "< 100 токенов", "Разрывов code",
                "Разрывов таблиц", "Разрывов предложений", "Дублирование"],
               structure_rows(report), ["---"] + ["---:"] * 8),
        "", "## Факты", "", *("- " + line for line in findings(report)),
        "", "## Примеры расхождений с baseline", "", *examples(report),
        "", "## Ограничения", "",
        "RU и EN одной темы зависимы, поэтому значимость считается по темам (sign test). "
        "Recall требует полного покрытия размеченного фрагмента; пересечение с разделом не засчитывается. "
        "Cosine не является вероятностью. Разрывы предложений считаются эвристикой по пунктуации.", "",
        "Разрезы RU/EN и по типам вопросов, ресурсы и выдача по каждому вопросу: `comparison.json`.", "",
    ]
    return "\n".join(lines)
