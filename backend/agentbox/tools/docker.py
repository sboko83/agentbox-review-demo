"""Docker-backed implementation of the workspace tool port.

Each tool call receives a short-lived container.  The only host mount is the
chosen workspace; network, capabilities and writable container layers are
removed.  The agent core still sees the same ``ToolExecutor`` contract as it
does with ``LocalExecutor``.
"""

from __future__ import annotations

import asyncio
import os
import shlex
import time
from dataclasses import dataclass
from pathlib import Path

from agentbox.core.ports import ToolCall, ToolResult
from agentbox.tools.local import (
    ALLOWED_COMMANDS,
    MAX_COMMAND_CHARS,
    MAX_FILE_CHARS,
    LocalExecutor,
    ToolFailure,
    _cap,
    _message,
)


@dataclass(frozen=True, slots=True)
class DockerLimits:
    """Limits applied to every short-lived tool container."""

    cpus: float = 1.0
    memory_mb: int = 512
    pids: int = 128

    def __post_init__(self) -> None:
        if self.cpus <= 0 or self.memory_mb < 64 or self.pids < 16:
            raise ValueError("Лимиты Docker должны быть положительными (RAM не меньше 64 MiB)")


class DockerExecutor(LocalExecutor):
    """Run all six tools inside an isolated Docker container.

    ``LocalExecutor`` is reused only for tool declarations, path validation and
    approval previews.  Filesystem reads and writes in ``run`` happen through
    Docker, never in the API process.
    """

    def __init__(
        self,
        root: Path,
        *,
        image: str,
        limits: DockerLimits,
        timeout: float = 30.0,
        allowed_commands: frozenset[str] = ALLOWED_COMMANDS,
        docker_bin: str = "docker",
    ) -> None:
        super().__init__(root, timeout=timeout, allowed_commands=allowed_commands)
        if not image.strip():
            raise ValueError("Не указан Docker-образ executor'а")
        self.image = image
        self.limits = limits
        self.docker_bin = docker_bin

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
        except OSError as error:
            output, exit_code, ok = f"ошибка Docker: {error}", None, False
        return ToolResult(
            call_id=call.id,
            output=output,
            ok=ok,
            exit_code=exit_code,
            duration_ms=int((time.monotonic() - started) * 1000),
        )

    def _docker_argv(self, argv: list[str]) -> list[str]:
        # The workspace remains writable because that is the requested coding
        # surface. Everything else is read-only, without network or Linux
        # capabilities; uid/gid prevent root-owned files in a Linux bind mount.
        return [
            self.docker_bin,
            "run",
            "--rm",
            "--init",
            "--network",
            "none",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=64m",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--cpus",
            str(self.limits.cpus),
            "--memory",
            f"{self.limits.memory_mb}m",
            "--pids-limit",
            str(self.limits.pids),
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "--mount",
            f"type=bind,src={self.root},dst=/workspace",
            "--workdir",
            "/workspace",
            self.image,
            *argv,
        ]

    async def _container(self, argv: list[str]) -> tuple[str, int]:
        try:
            process = await asyncio.create_subprocess_exec(
                *self._docker_argv(argv),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
        except FileNotFoundError as error:
            raise ToolFailure("Docker не найден: установите и запустите Docker Desktop") from error
        try:
            stdout, _ = await asyncio.wait_for(process.communicate(), self.timeout)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            raise ToolFailure(f"контейнер не уложился в {self.timeout:g} с") from None
        output = _cap(stdout.decode("utf-8", errors="replace"), MAX_COMMAND_CHARS)
        return output.rstrip(), process.returncode

    def _relative(self, raw: str) -> str:
        return self.relative(self.resolve(raw))

    async def _python(self, source: str, *args: str) -> tuple[str, int]:
        return await self._container(["python3", "-c", source, *args])

    async def _tool_read_file(self, args: dict) -> tuple[str, int | None]:
        path = self._relative(args["path"])
        output, code = await self._python(
            "from pathlib import Path; import sys; print(Path(sys.argv[1]).read_text(encoding='utf-8'), end='')",
            path,
        )
        return _cap(output, MAX_FILE_CHARS), None if code == 0 else code

    async def _tool_list_dir(self, args: dict) -> tuple[str, int | None]:
        path = self._relative(args.get("path") or ".")
        source = (
            "from pathlib import Path; import sys; "
            "skip={'.git','node_modules','.venv','__pycache__','dist','.pytest_cache'}; "
            "p=Path(sys.argv[1]); "
            "items=[f'{x.name}{\"/\" if x.is_dir() else \"\"}{\"\" if x.is_dir() else f\"  {x.stat().st_size} B\"}' "
            "for x in sorted(p.iterdir(), key=lambda x:(x.is_file(), x.name)) if x.name not in skip]; "
            "print('\\n'.join(items) or '(пусто)')"
        )
        output, code = await self._python(source, path)
        return output, None if code == 0 else code

    async def _tool_grep(self, args: dict) -> tuple[str, int | None]:
        path = self._relative(args.get("path") or ".")
        source = (
            "from pathlib import Path; import fnmatch,re,sys; "
            "root=Path(sys.argv[1]); pattern=re.compile(sys.argv[2]); mask=sys.argv[3]; skip={'.git','node_modules','.venv','__pycache__','dist','.pytest_cache'}; found=[]; "
            "files=[root] if root.is_file() else root.rglob(mask); "
            "exec(\"for file in files:\\n"
            "  if not file.is_file() or skip & set(file.parts): continue\\n"
            "  try: lines=file.read_text(encoding='utf-8').splitlines()\\n"
            "  except (OSError,UnicodeDecodeError): continue\\n"
            "  for number,line in enumerate(lines,1):\\n"
            "    if pattern.search(line): found.append(f'{file}:{number}: {line.strip()[:200]}')\\n"
            "    if len(found)>=200: break\\n"
            "  if len(found)>=200: break\"); "
            "print('\\n'.join(found) if found else 'совпадений нет')"
        )
        output, code = await self._python(source, path, args["pattern"], args.get("glob") or "*")
        return output, None if code == 0 else code

    async def _tool_write_file(self, args: dict) -> tuple[str, int | None]:
        path = self._relative(args["path"])
        # Content travels as an argv value: no shell, no host-side temporary
        # file, and every write lands inside the bind mount.
        source = (
            "from pathlib import Path; import sys; p=Path(sys.argv[1]); "
            "existed=p.is_file(); p.parent.mkdir(parents=True,exist_ok=True); "
            "content=sys.argv[2]; p.write_text(content,encoding='utf-8'); "
            "print(('перезаписан' if existed else 'создан') + f' {p}, {len(content)} символов')"
        )
        output, code = await self._python(source, path, args["content"])
        return output, None if code == 0 else code

    async def _tool_edit_file(self, args: dict) -> tuple[str, int | None]:
        path = self._relative(args["path"])
        source = (
            "from pathlib import Path; import sys; p=Path(sys.argv[1]); text=p.read_text(encoding='utf-8'); old=sys.argv[2]; new=sys.argv[3]; count=text.count(old); "
            "exec(\"if count == 0: raise ValueError('фрагмент old_text в файле не найден')\\n"
            "if count > 1: raise ValueError(f'фрагмент old_text встречается {count} раз — уточните его')\"); "
            "p.write_text(text.replace(old,new,1),encoding='utf-8'); print(f'правка применена к {p}')"
        )
        output, code = await self._python(source, path, args["old_text"], args["new_text"])
        return output, None if code == 0 else code

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
        output, code = await self._container(argv)
        return f"код выхода {code}\n{output}".rstrip(), code
