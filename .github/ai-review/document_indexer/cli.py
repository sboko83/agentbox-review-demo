"""Thin CLI over the public Python API."""
import argparse
import json
import sys
import sqlite3
import textwrap
from dataclasses import asdict
from . import load_config, build_index, search_index, compare_strategies


def human_output(command, result):
    if command == "build":
        config = result["config"]
        print(f"Built {config['strategy']}: {result['documents']} documents, {result['chunks']} chunks")
        print(f"Embeddings: {result['embeddings_computed']} computed, {result['embeddings_cached']} from cache; "
              f"{result['dimension']} dimensions, float32, L2 normalized")
        print(f"Model load: {result['model_load_seconds']:.2f}s; embeddings: {result['embedding_seconds']:.2f}s; "
              f"total processing: {result['processing_seconds']:.2f}s")
        print(f"Index: {config['output']}")
    elif command == "search":
        for rank, item in enumerate(result, 1):
            chunk = item["chunk"]
            print(f"{rank}. cosine={item['score']:.4f}  {chunk['file']}:{chunk['start_line']}-{chunk['end_line']}")
            print("   " + (" | ".join(chunk["section_path"]) or chunk["title"]))
            print(textwrap.fill(" ".join(chunk["text"].split())[:250], width=88))
    else:
        from .evaluation import retrieval_rows, findings
        print(f"{'Variant':20} {'chunks':>6} {'Hit@5':>6} {'Rec@5':>6} {'MRR@5':>6}  topics +/-/= vs baseline")
        for key, chunks, hit, recall, mrr, _, versus in retrieval_rows(result):
            print(f"{key:20} {chunks:6} {hit:>6} {recall:>6} {mrr:>6}  {versus}")
        print("Dev-selected: " + ", ".join(result["selected"].values()))
        for line in findings(result):
            print(textwrap.fill(line, width=110, subsequent_indent="  "))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="document-indexer")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--config", required=True)
    for name in ("root", "output", "cache", "strategy", "device"):
        build.add_argument("--" + name)
    for name in ("target-tokens", "overlap-tokens", "batch-size"):
        build.add_argument("--" + name, type=int)
    build.add_argument("--no-embedding-cache", dest="embedding_cache", action="store_const", const=False,
                       help="Encode every chunk even if its vector is cached")
    search = commands.add_parser("search")
    search.add_argument("--index", required=True)
    search.add_argument("--query", required=True)
    search.add_argument("--k", type=int, default=5)
    compare = commands.add_parser("compare")
    compare.add_argument("--config", required=True)
    compare.add_argument("--questions")
    compare.add_argument("--output", required=True)
    compare.add_argument("--reuse", action="store_true",
                         help="Evaluate matching saved indexes without rebuilding")
    for command in (build, search, compare):
        command.add_argument("--json", action="store_true")
    args = vars(parser.parse_args(argv))
    command, json_output = args.pop("command"), args.pop("json")
    try:
        if command == "build":
            result = build_index(load_config(args.pop("config"), **args))
        elif command == "search":
            result = [asdict(r) for r in search_index(**args)]
        else:
            config = load_config(args.pop("config"))
            progress = None if json_output else lambda message: print(message, flush=True)
            result = compare_strategies(config, progress=progress, **args)
        if json_output:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            human_output(command, result)
        return 0
    except (ValueError, OSError, RuntimeError, KeyError, TypeError, sqlite3.DatabaseError) as exc:
        message = json.dumps({"error": str(exc)}, ensure_ascii=False) if json_output else f"Error: {exc}"
        print(message, file=sys.stderr)
        return 1
