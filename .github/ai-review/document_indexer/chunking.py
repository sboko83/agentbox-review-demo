"""Lossless source ranges with model-tokenizer budgets."""
import json
import re
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from .documents import digest
from .types import Chunk

CHUNKING_VERSION = 2
MODEL_TOKEN_LIMIT = 512
# Whole sections are packed while they share this many leading headings.
# Depth 1 (the page title) removes tiny heading-only chunks; see THEORY.md.
PACK_DEPTH = 1
# Language-independent punctuation heuristic, not a linguistic parser.
SENTENCE_END = r'[.!?。！？]["”’\x27)]*(?:\s+|$)'
RECURSIVE_SEPARATORS = (r"\n[ \t]*\n+", r"\n", SENTENCE_END)
FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
HEADING = re.compile(r"^ {0,3}(#{1,6})\s+(.+?)\s*#*\s*$")
LIST_ITEM = re.compile(r"^\s*(?:[-+*]|\d+[.)])\s")


@dataclass(frozen=True)
class Block:
    start: int
    end: int
    section: tuple[str, ...]
    section_id: str
    kind: str


def parse_blocks(document):
    """Split Markdown into headings, paragraphs, lists, tables and fenced code."""
    text = document.text
    markdown = document.file.lower().endswith(".md")
    blocks, stack = [], []
    offset, start, kind, fence, section_id = 0, 0, "paragraph", None, "0"
    section = ()

    def flush(end):
        nonlocal start
        if end > start:
            blocks.append(Block(start, end, section, section_id, kind))
        start = end

    for line in text.splitlines(keepends=True):
        if fence:
            offset += len(line)
            closing = r"^ {0,3}" + re.escape(fence[0]) + "{" + str(len(fence)) + r",}\s*$"
            if re.match(closing, line):
                fence = None
                flush(offset)
                kind = "paragraph"
            continue
        marker = FENCE.match(line) if markdown else None
        heading = HEADING.match(line) if markdown else None
        if marker:
            flush(offset)
            kind, fence = "code", marker[1]
        elif heading:
            flush(offset)
            level = len(heading[1])
            stack = [(n, name) for n, name in stack if n < level] + [(level, heading[2])]
            section = tuple(name for _, name in stack)
            section_id, kind = str(offset), "heading"
        elif not line.strip():
            offset += len(line)
            flush(offset)
            kind = "paragraph"
            continue
        else:
            if "|" in line:
                next_kind = "table"
            elif LIST_ITEM.match(line):
                next_kind = "list"
            else:
                next_kind = "paragraph"
            table_edge = next_kind != kind and "table" in (kind, next_kind)
            if kind == "heading" or table_edge:
                flush(offset)
            kind = next_kind
        offset += len(line)
    flush(len(text))
    return blocks


def token_count(tokenizer, text, special=False):
    encoded = tokenizer(text, add_special_tokens=special, truncation=False, verbose=False)
    return len(encoded["input_ids"])


def make_chunks(document, config, tokenizer):
    return _Chunker(document, config, tokenizer).run()


def _between(values, low, high):
    """Sorted values v with low < v <= high."""
    return values[bisect_right(values, low):bisect_right(values, high)]


class _Chunker:
    """Greedy token-budget windows; strategies differ in groups and boundaries.

    fixed      one group per document, any token end is a valid boundary.
    structure  whole sections packed per parent section; splits only oversized ones.
    sentence   sentence ends are preferred boundaries.
    recursive  paragraph -> line -> sentence ends, whichever first fits.
    """

    def __init__(self, document, config, tokenizer):
        self.document, self.config, self.tokenizer = document, config, tokenizer
        self.text = document.text
        self.blocks = parse_blocks(document)
        offsets = tokenizer(self.text, add_special_tokens=False, return_offsets_mapping=True,
                            truncation=False, verbose=False)["offset_mapping"]
        self.offsets = sorted(offsets)
        self.token_starts = [start for start, _ in self.offsets]
        self.token_ends = sorted({end for _, end in offsets})
        self.block_ends = sorted({block.end for block in self.blocks})
        self.sentence_ends = sorted({m.end() for m in re.finditer(SENTENCE_END, self.text)})
        self.preferred = []
        if config.strategy == "sentence":
            self.preferred = sorted(set(self.sentence_ends) | {len(self.text)})
        elif config.strategy == "recursive":
            found = set()
            self._split_recursive(0, len(self.text), 0, found)
            self.preferred = sorted(found)

    # Budget and context -------------------------------------------------

    def tokens(self, start, end):
        return token_count(self.tokenizer, self.text[start:end])

    def details(self, start, end):
        touched = [b for b in self.blocks if b.start < end and b.end > start]
        paths = list(dict.fromkeys(" / ".join(b.section) for b in touched if b.section))
        context = self.document.title + ("\n" + " | ".join(paths) if paths else "")
        embedding_text = "passage: " + context + "\n\n" + self.text[start:end]
        return touched, paths, context, embedding_text

    def fits(self, start, end):
        if self.tokens(start, end) > self.config.target_tokens:
            return False
        embedding_text = self.details(start, end)[3]
        return token_count(self.tokenizer, embedding_text, True) <= MODEL_TOKEN_LIMIT

    def largest_end(self, start, limit):
        """Farthest token/preferred boundary in (start, limit] that fits the budget."""
        candidates = sorted(set(_between(self.token_ends, start, limit)) | {limit}
                            | set(_between(self.preferred, start, limit)))
        low, high, best = 0, len(candidates) - 1, None
        while low <= high:
            middle = (low + high) // 2
            if self.fits(start, candidates[middle]):
                best, low = candidates[middle], middle + 1
            else:
                high = middle - 1
        return best

    # Strategy inputs ----------------------------------------------------

    def _split_recursive(self, start, end, level, found):
        if level == len(RECURSIVE_SEPARATORS) or self.fits(start, end):
            found.add(end)
            return
        piece = self.text[start:end]
        edges = [start] + [start + m.end() for m in re.finditer(RECURSIVE_SEPARATORS[level], piece)]
        for left, right in zip(edges, edges[1:] + [end]):
            if left < right:
                self._split_recursive(left, right, level + 1, found)

    def groups(self):
        if self.config.strategy != "structure":
            return [(0, len(self.text))]
        sections = []
        for block in self.blocks:
            if sections and sections[-1][2] == block.section_id:
                sections[-1][1] = block.end
            else:
                sections.append([block.start, block.end, block.section_id, block.section[:PACK_DEPTH]])
        # Small neighbouring sections share a chunk; oversized ones are split below.
        groups = []
        for start, end, _, parent in sections:
            if groups and groups[-1][2] == parent and self.fits(groups[-1][0], end):
                groups[-1][1] = end
            else:
                groups.append([start, end, parent])
        return [(start, end) for start, end, _ in groups]

    # Main loop ----------------------------------------------------------

    def run(self):
        chunks = []
        for group_start, group_end in self.groups():
            self._split_group(group_start, group_end, chunks)
        return chunks

    def _split_group(self, group_start, group_end, chunks):
        structure = self.config.strategy == "structure"
        start = previous_end = group_start
        while start < group_end:
            end = self.largest_end(start, group_end)
            if end is None:
                raise ValueError(f"Title/section leaves no token budget: {self.document.file}:{start}")
            fresh = max(start, previous_end)
            if structure and start < previous_end and not _between(self.block_ends, previous_end, end):
                # Drop optional overlap before splitting an otherwise intact block.
                start = previous_end
                continue
            reason = None
            if self.preferred and end < group_end:
                natural = _between(self.preferred, fresh, end)
                if natural:
                    end = natural[-1]
                elif start < previous_end:
                    # Optional overlap must not force a split of the next unit.
                    start = previous_end
                    continue
                elif self.config.strategy == "sentence":
                    reason = "oversized_sentence"
                else:
                    reason = "recursive_token_fallback"
            if structure and end < group_end:
                end, reason = self._structure_end(start, fresh, end)
            if end <= previous_end:
                start = previous_end
                continue
            if not self.text[start:end].strip():
                start = end
                continue
            chunks.append(self._chunk(start, end, previous_end, reason))
            if end == group_end:
                break
            previous_end, start = end, self._overlap_start(start, end)

    def _structure_end(self, start, fresh, end):
        """Cut at the last whole block; fall back to a sentence inside prose."""
        whole = _between(self.block_ends, fresh, end)
        if whole:
            return whole[-1], None
        atomic = next((b for b in self.blocks if b.start <= start < b.end), None)
        sentences = _between(self.sentence_ends, fresh, end)
        if sentences and atomic and atomic.kind not in ("code", "table"):
            return sentences[-1], "sentence"
        return end, "oversized_" + (atomic.kind if atomic else "block")

    def _overlap_start(self, start, end):
        limit = self.config.overlap_tokens
        if not limit:
            return end
        if self.config.strategy == "structure":
            # Carry whole trailing blocks only; never a partial block.
            touched = self.details(start, end)[0]
            eligible = [b.start for b in touched
                        if start < b.start < end and self.tokens(b.start, end) <= limit]
            return min(eligible, default=end)
        if self.preferred:
            # Retain whole sentences/recursive units, not token fragments.
            eligible = [b for b in _between(self.preferred, start, end - 1)
                        if self.tokens(b, end) <= limit]
            return min(eligible, default=end)
        first = bisect_right(self.token_starts, start)
        last = bisect_left(self.token_starts, end)
        inside = [s for s, e in self.offsets[first:last] if e <= end]
        next_start = end
        for candidate in reversed(inside):
            if self.tokens(candidate, end) > limit:
                break
            next_start = candidate
        return next_start

    def _chunk(self, start, end, previous_end, reason):
        document, config = self.document, self.config
        touched, paths, context, embedding_text = self.details(start, end)
        overlap = self.tokens(start, min(previous_end, end)) if start < previous_end else 0
        identity = json.dumps([document.source_id, document.content_hash, CHUNKING_VERSION,
                               config.strategy, config.target_tokens, config.overlap_tokens,
                               config.model, config.model_revision, start, end])
        return Chunk(
            chunk_id=digest(identity),
            source_id=document.source_id,
            file=document.file,
            text=self.text[start:end],
            embedding_text=embedding_text,
            context=context,
            start=start,
            end=end,
            start_line=self.text.count("\n", 0, start) + 1,
            end_line=self.text.count("\n", 0, end - 1) + 1,
            section_path=paths,
            section_ids=list(dict.fromkeys(b.section_id for b in touched)),
            tokens=token_count(self.tokenizer, embedding_text, True),
            overlap_tokens=overlap,
            split_reason=reason,
            title=document.title,
            source=document.source,
        )
