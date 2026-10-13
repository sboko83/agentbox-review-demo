"""Events an agent emits while answering.

The API layer turns these into SSE frames; tests consume them directly. The
agent never formats HTTP itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(slots=True)
class Delta:
    """A piece of the answer text."""

    text: str
    type: Literal["delta"] = "delta"


@dataclass(slots=True)
class ContextPrepared:
    """The exact context view selected for the following provider call."""

    view: dict = field(default_factory=dict)
    type: Literal["context"] = "context"


@dataclass(slots=True)
class InvariantChecked:
    """A semantic gate finished before anything it refused became observable."""

    target: Literal["request", "response", "tool"]
    outcome: Literal["allow", "conflict", "error"]
    rule_ids: list[str] = field(default_factory=list)
    explanation: str = ""
    alternative: str | None = None
    model: str = ""
    usage: dict = field(default_factory=dict)
    input_estimate_tokens: int = 0
    latency_ms: int = 0
    type: Literal["invariant_check"] = "invariant_check"


@dataclass(slots=True)
class ApprovalNeeded:
    """A mutating call is waiting for a human in `ask` mode."""

    call_id: str
    tool: str
    arguments: str
    preview: str
    type: Literal["approval"] = "approval"


@dataclass(slots=True)
class ToolStarted:
    """A tool call passed the checks and is running."""

    call_id: str
    tool: str
    arguments: str
    preview: str
    type: Literal["tool"] = "tool"


@dataclass(slots=True)
class ToolFinished:
    """The tool call is over; `output` is what the model will see."""

    call_id: str
    tool: str
    output: str
    ok: bool
    exit_code: int | None = None
    duration_ms: int = 0
    type: Literal["tool_result"] = "tool_result"


@dataclass(slots=True)
class UsageReported:
    """Measured turn cost plus the estimate that preceded the provider call."""

    estimate: dict = field(default_factory=dict)
    actual: dict = field(default_factory=dict)
    context: dict = field(default_factory=dict)
    ledger: dict = field(default_factory=dict)
    type: Literal["usage"] = "usage"


@dataclass(slots=True)
class Done:
    """The answer is complete; carries spend and memory state."""

    text: str
    usage: dict = field(default_factory=dict)
    ledger: dict = field(default_factory=dict)
    memory_tokens: int = 0
    dropped: int = 0
    steps: int = 1
    type: Literal["done"] = "done"


@dataclass(slots=True)
class Failed:
    """The call failed; the message is meant for the user."""

    message: str
    type: Literal["error"] = "error"


AgentEvent = (
    ContextPrepared
    | InvariantChecked
    | Delta
    | ApprovalNeeded
    | ToolStarted
    | ToolFinished
    | UsageReported
    | Done
    | Failed
)
