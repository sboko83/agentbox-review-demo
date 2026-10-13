"""Local executor: the six tools, run in a subprocess-backed workspace.

The workspace root is the security boundary. Every path argument is resolved
and checked against it before anything touches the filesystem, so `../../etc`
and a symlink pointing outside are both refused here rather than inside a tool.

This is not real isolation: the process can technically see the whole
filesystem, and the guarantee rests on these checks. Phase 6 replaces this
class with a Docker-backed implementation of the same port; nothing above it
changes.
"""

from __future__ import annotations

import asyncio
import difflib
import re
import shlex
import time
from pathlib import Path

from agentbox.core.ports import ToolCall, ToolResult, ToolSpec

MAX_FILE_CHARS = 20_000
MAX_COMMAND_CHARS = 10_000
MAX_GREP_MATCHES = 200
MAX_DIFF_LINES = 200
SKIP_DIRS = {".git", "node_modules", ".venv", "__pycache__", "dist", ".pytest_cache"}

# Shell metacharacters are not interpreted: the command is split with shlex and
# run without a shell, which is the only thing that makes an allowlist mean
# anything. Pipes and redirects therefore do not work — that is the price.
ALLOWED_COMMANDS = frozenset(
    {
        "awk", "cat", "cp", "diff", "echo", "find", "git", "grep", "head", "ls",
        "make", "mkdir", "mv", "node", "npm", "npx", "pip", "pip3", "pwd",
        "python", "python3", "pytest", "sed", "sort", "tail", "touch", "tree",
        "uniq", "wc", "which",
    }
)

_STRING = {"type": "string"}


TOOLS: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="read_file",
        description="Прочитать текстовый файл из рабочей папки.",
        parameters={
            "type": "object",
            "properties": {"path": {**_STRING, "description": "Путь от корня папки."}},
            "required": ["path"],
            "additionalProperties": False,
        },
    ),
    ToolSpec(
        name="list_dir",
        description="Показать содержимое каталога в рабочей папке.",
        parameters={
            "type": "object",
            "properties": {"path": {**_STRING, "description": "Каталог, по умолчанию корень."}},
            "required": [],
            "additionalProperties": False,
        },
    ),
    ToolSpec(
        name="grep",
        description="Найти регулярное выражение в файлах рабочей папки.",
        parameters={
            "type": "object",
            "properties": {
                "pattern": {**_STRING, "description": "Регулярное выражение Python."},
                "path": {**_STRING, "description": "Где искать, по умолчанию корень."},
                "glob": {**_STRING, "description": "Маска имён, например *.py."},
            },
            "required": ["pattern"],
            "additionalProperties": False,
        },
    ),
    ToolSpec(
        name="write_file",
        description="Создать файл или полностью переписать его содержимое.",
        parameters={
            "type": "object",
            "properties": {
                "path": _STRING,
                "content": {**_STRING, "description": "Новое содержимое целиком."},
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        },
        mutating=True,
    ),
    ToolSpec(
        name="edit_file",
        description="Заменить в файле точный фрагмент текста на другой.",
        parameters={
            "type": "object",
            "properties": {
                "path": _STRING,
                "old_text": {**_STRING, "description": "Фрагмент, встречающийся один раз."},
                "new_text": {**_STRING, "description": "Чем заменить."},
            },
            "required": ["path", "old_text", "new_text"],
            "additionalProperties": False,
        },
        mutating=True,
    ),
    ToolSpec(
        name="run_command",
        description=(
            "Выполнить команду в рабочей папке без shell. Конвейеры и "
            "перенаправления не работают. Разрешены: " + ", ".join(sorted(ALLOWED_COMMANDS))
        ),
        parameters={
            "type": "object",
            "properties": {"command": {**_STRING, "description": "Команда с аргументами."}},
            "required": ["command"],
            "additionalProperties": False,
        },
        mutating=True,
    ),
)


class ToolFailure(Exception):
    """The call is wrong or forbidden; the model gets the text as the result."""


class LocalExecutor:
    """Runs the tools against a local directory."""

    def __init__(
        self,
        root: Path,
        *,
        timeout: float = 30.0,
        allowed_commands: frozenset[str] = ALLOWED_COMMANDS,
    ) -> None:
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise ValueError(f"рабочая папка не существует: {self.root}")
        self.timeout = timeout
        self.allowed_commands = allowed_commands

    def describe(self) -> list[ToolSpec]:
        return list(TOOLS)

    # --- boundary ---------------------------------------------------------

    def resolve(self, raw: str) -> Path:
        """Resolve a tool path inside the workspace, or refuse it."""
        candidate = Path(raw)
        if candidate.is_absolute():
            resolved = candidate.resolve()
        else:
            resolved = (self.root / candidate).resolve()
        if resolved != self.root and self.root not in resolved.parents:
            raise ToolFailure(
                f"путь {raw} выходит за пределы рабочей папки {self.root}"
            )
        return resolved

    def relative(self, path: Path) -> str:
        return str(path.relative_to(self.root)) if path != self.root else "."

    # --- approval preview -------------------------------------------------

    def preview(self, call: ToolCall) -> str:
        """A one-glance summary of what the call would do."""
        try:
            args = call.arguments()
        except ValueError as error:
            return str(error)
        try:
            if call.tool == "run_command":
                return f"$ {args.get('command', '')}"
            if call.tool == "write_file":
                path = self.resolve(args["path"])
                before = path.read_text(encoding="utf-8") if path.is_file() else ""
                return _diff(self.relative(path), before, args.get("content", ""))
            if call.tool == "edit_file":
                path = self.resolve(args["path"])
                before = path.read_text(encoding="utf-8")
                after = _replace_once(before, args["old_text"], args["new_text"])
                return _diff(self.relative(path), before, after)
            if "path" in args:
                return self.relative(self.resolve(args["path"]))
        except (ToolFailure, KeyError, OSError, UnicodeDecodeError) as error:
            return str(error)
        return call.arguments_json

    # --- dispatch ---------------------------------------------------------

    async def run(self, call: ToolCall) -> ToolResult:
        started = time.monotonic()
        try:
            args = call.arguments()
            handler = getattr(self, f"_tool_{call.tool}", None)
            if handler is None:
                raise ToolFailure(f"инструмент {call.tool} не существует")
            output, exit_code = await handler(args)
            ok = exit_code in (None, 0)
        except (ToolFailure, KeyError, ValueError) as error:
            output, exit_code, ok = _message(error), None, False
        except (OSError, UnicodeDecodeError) as error:
            output, exit_code, ok = f"ошибка файловой системы: {error}", None, False
        return ToolResult(
            call_id=call.id,
            output=output,
            ok=ok,
            exit_code=exit_code,
            duration_ms=int((time.monotonic() - started) * 1000),
        )

    # --- tools ------------------------------------------------------------

    async def _tool_read_file(self, args: dict) -> tuple[str, int | None]:
        path = self.resolve(args["path"])
        if not path.is_file():
            raise ToolFailure(f"файл не найден: {args['path']}")
        text = await asyncio.to_thread(path.read_text, encoding="utf-8")
        return _cap(text, MAX_FILE_CHARS), None

    async def _tool_list_dir(self, args: dict) -> tuple[str, int | None]:
        path = self.resolve(args.get("path") or ".")
        if not path.is_dir():
            raise ToolFailure(f"каталог не найден: {args.get('path') or '.'}")
        lines = []
        for entry in sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name)):
            if entry.name in SKIP_DIRS:
                continue
            mark = "/" if entry.is_dir() else ""
            size = "" if entry.is_dir() else f"  {entry.stat().st_size} B"
            lines.append(f"{entry.name}{mark}{size}")
        return "\n".join(lines) or "(пусто)", None

    async def _tool_grep(self, args: dict) -> tuple[str, int | None]:
        try:
            pattern = re.compile(args["pattern"])
        except re.error as error:
            raise ToolFailure(f"плохое регулярное выражение: {error}") from error
        root = self.resolve(args.get("path") or ".")
        mask = args.get("glob") or "*"
        matches = await asyncio.to_thread(self._search, root, pattern, mask)
        if not matches:
            return "совпадений нет", None
        return "\n".join(matches), None

    def _search(self, root: Path, pattern: re.Pattern[str], mask: str) -> list[str]:
        files = [root] if root.is_file() else sorted(root.rglob(mask))
        found: list[str] = []
        for file in files:
            if not file.is_file() or SKIP_DIRS & set(file.parts):
                continue
            try:
                text = file.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            for number, line in enumerate(text.splitlines(), start=1):
                if pattern.search(line):
                    found.append(f"{self.relative(file)}:{number}: {line.strip()[:200]}")
                    if len(found) >= MAX_GREP_MATCHES:
                        found.append(f"… обрезано на {MAX_GREP_MATCHES} совпадениях")
                        return found
        return found

    async def _tool_write_file(self, args: dict) -> tuple[str, int | None]:
        path = self.resolve(args["path"])
        content = args["content"]
        path.parent.mkdir(parents=True, exist_ok=True)
        existed = path.is_file()
        await asyncio.to_thread(path.write_text, content, encoding="utf-8")
        verb = "перезаписан" if existed else "создан"
        return f"{verb} {self.relative(path)}, {len(content)} символов", None

    async def _tool_edit_file(self, args: dict) -> tuple[str, int | None]:
        path = self.resolve(args["path"])
        if not path.is_file():
            raise ToolFailure(f"файл не найден: {args['path']}")
        before = await asyncio.to_thread(path.read_text, encoding="utf-8")
        after = _replace_once(before, args["old_text"], args["new_text"])
        await asyncio.to_thread(path.write_text, after, encoding="utf-8")
        return f"правка применена к {self.relative(path)}", None

    async def _tool_run_command(self, args: dict) -> tuple[str, int | None]:
        line = (args.get("command") or "").strip()
        if not line:
            raise ToolFailure("пустая команда")
        try:
            argv = shlex.split(line)
        except ValueError as error:
            raise ToolFailure(f"команда не разбирается: {error}") from error
        if argv[0] not in self.allowed_commands:
            raise ToolFailure(
                f"команда {argv[0]} не в списке разрешённых: "
                + ", ".join(sorted(self.allowed_commands))
            )

        # An allowlisted command may still be absent on this machine; that is a
        # tool failure the model can react to, not a filesystem error.
        try:
            process = await asyncio.create_subprocess_exec(
                *argv,
                cwd=self.root,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
        except FileNotFoundError as error:
            raise ToolFailure(f"исполняемый файл не найден: {argv[0]}") from error
        try:
            stdout, _ = await asyncio.wait_for(process.communicate(), self.timeout)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            raise ToolFailure(f"команда не уложилась в {self.timeout:g} с") from None

        text = _cap(stdout.decode("utf-8", errors="replace"), MAX_COMMAND_CHARS)
        return f"код выхода {process.returncode}\n{text}".rstrip(), process.returncode


def _replace_once(text: str, old: str, new: str) -> str:
    occurrences = text.count(old)
    if occurrences == 0:
        raise ToolFailure("фрагмент old_text в файле не найден")
    if occurrences > 1:
        raise ToolFailure(
            f"фрагмент old_text встречается {occurrences} раз — уточните его"
        )
    return text.replace(old, new, 1)


def _diff(name: str, before: str, after: str) -> str:
    lines = list(
        difflib.unified_diff(
            before.splitlines(),
            after.splitlines(),
            fromfile=f"a/{name}",
            tofile=f"b/{name}",
            lineterm="",
            n=2,
        )
    )
    if not lines:
        return f"{name}: изменений нет"
    if len(lines) > MAX_DIFF_LINES:
        lines = lines[:MAX_DIFF_LINES] + [f"… ещё {len(lines) - MAX_DIFF_LINES} строк"]
    return "\n".join(lines)


def _cap(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n… обрезано, всего {len(text)} символов"


def _message(error: Exception) -> str:
    if isinstance(error, KeyError):
        return f"не хватает обязательного аргумента {error}"
    return str(error)
