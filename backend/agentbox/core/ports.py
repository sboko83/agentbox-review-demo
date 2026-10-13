"""Ports the agent core depends on.

The agent itself performs no I/O: everything outside is injected as one of
these protocols. That is what makes spawning many agents inside a single
process cheap, and what lets tests run the whole pipeline without a network.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import AsyncIterator, Protocol


@dataclass(slots=True)
class Usage:
    """Cost of a single model call."""

    tokens_in: int = 0
    tokens_out: int = 0
    reasoning: int = 0
    cost: float = 0.0


@dataclass(slots=True)
class ToolCall:
    """A function call the model asked for.

    Arguments stay as the raw JSON string the model produced: it is what has to
    be echoed back to the provider verbatim on the next turn, and a model that
    emits broken JSON must get a tool error rather than crash the loop.
    """

    id: str
    tool: str
    arguments_json: str = "{}"

    def arguments(self) -> dict:
        """Parsed arguments. Raises `ValueError` on malformed JSON."""
        try:
            value = json.loads(self.arguments_json or "{}")
        except json.JSONDecodeError as error:
            raise ValueError(f"аргументы не разбираются как JSON: {error}") from error
        if not isinstance(value, dict):
            raise ValueError("аргументы инструмента должны быть объектом")
        return value


@dataclass(slots=True)
class ToolResult:
    """What the executor produced; `output` goes back into the dialogue."""

    call_id: str
    output: str
    ok: bool = True
    exit_code: int | None = None
    duration_ms: int = 0


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """Declaration of one tool, in the shape the provider expects.

    `mutating` is what the confirmation modes key on: read-only sessions never
    see these tools, `ask` sessions stop for approval before running them.
    """

    name: str
    description: str
    parameters: dict
    mutating: bool = False

    def as_function(self) -> dict:
        """Declaration shape for /responses: the function is the object itself."""
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }

    def as_chat_function(self) -> dict:
        """Declaration shape for /chat/completions: the function is nested."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass(slots=True)
class Chunk:
    """A piece of the stream: text, a finished tool call, or the usage totals."""

    text: str = ""
    tool_call: ToolCall | None = None
    usage: Usage | None = None


@dataclass(slots=True)
class LLMRequest:
    """One provider call, in vendor-neutral terms.

    `messages` is already rendered in the transport's own history format — the
    agent asks its `Conversation` for the right one. Sampling fields are stated
    the way the caller means them; the transport drops those the model refuses
    and renames the rest.
    """

    model: str
    messages: list[dict]
    temperature: float | None = None
    max_output_tokens: int | None = None
    text_format: dict | None = None
    tools: list[ToolSpec] = field(default_factory=list)
    top_p: float | None = None
    top_k: int | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    stop: tuple[str, ...] = ()
    seed: int | None = None
    reasoning_effort: str | None = None


class LLMPort(Protocol):
    """A model provider that streams an answer.

    The application owns the port's lifetime, so closing is part of the
    contract: an implementation holding a connection pool must release it.
    """

    def stream(self, request: LLMRequest) -> AsyncIterator[Chunk]:
        ...

    async def aclose(self) -> None:
        ...


class LLMError(RuntimeError):
    """The call failed and retrying it makes no sense."""


class ToolExecutor(Protocol):
    """Runs tool calls inside a workspace.

    The executor owns the workspace boundary and the process; it knows nothing
    about confirmation modes — the agent decides what may run, the executor
    decides how.
    """

    def describe(self) -> list[ToolSpec]:
        ...

    def preview(self, call: ToolCall) -> str:
        """What the user is asked to approve: a diff, a command line, a path."""

    async def run(self, call: ToolCall) -> ToolResult:
        ...


class ApprovalPort(Protocol):
    """Asks a human whether a mutating call may run.

    Only consulted in `ask` mode; a timeout or a closed browser must resolve to
    a refusal, never to a silent yes.
    """

    async def confirm(self, call: ToolCall, preview: str) -> bool:
        ...
