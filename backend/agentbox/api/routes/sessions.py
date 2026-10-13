"""HTTP surface for sessions, chat and tool approvals.

The adapter layer: it restores an agent from storage, streams its events to the
browser as SSE and writes the result back. The agent itself knows nothing about
any of this.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import sqlite3
from pathlib import Path
from typing import AsyncIterator, Literal

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, ValidationError, model_validator

from agentbox.core.agent import REFUSED, Agent
from agentbox.core.context import (
    LAYER_KINDS,
    MemoryResult,
    MemoryRouter,
    SummaryContextStrategy,
    build_context_view,
    strategy_for,
)
from agentbox.core.events import (
    Done,
    Failed,
    InvariantChecked,
    ToolFinished,
    ToolStarted,
    UsageReported,
)
from agentbox.core.ledger import BudgetExceeded, Ledger
from agentbox.core.invariants import InvariantRule, InvariantSet
from agentbox.core.ports import LLMError, ToolCall, ToolResult, Usage
from agentbox.core.spec import AgentSpec
from agentbox.core.task_machine import (
    STAGE_TEXT,
    TaskError,
    TaskState,
    available_transitions,
    stage_role,
)
from agentbox.core.task_tracker import TaskTracker, TrackerResult, tracker_runs
from agentbox.core.user_profile import UserProfile
from agentbox.store import repository as repo
from agentbox.tools.docker import DockerExecutor, DockerLimits
from agentbox.tools.local import LocalExecutor

router = APIRouter(prefix="/api/sessions", tags=["sessions"])
schema_router = APIRouter(prefix="/api", tags=["agent-spec"])

# How long a mutating call waits for a human before it counts as refused. Long
# enough to read a diff, short enough that a closed tab does not pin a request.
APPROVAL_TIMEOUT_S = 300.0


class CreateSession(BaseModel):
    spec: AgentSpec | None = None
    preset_id: str | None = None
    # Omitted means the default profile; an explicit null means none.
    user_profile_id: str | None = None
    title: str | None = None
    workspace_path: str | None = None
    executor_kind: Literal["local", "docker"] | None = None

    @model_validator(mode="after")
    def one_spec_source(self) -> "CreateSession":
        if self.spec is not None and self.preset_id is not None:
            raise ValueError("Передайте либо spec, либо preset_id")
        return self


class SendMessage(BaseModel):
    text: str = Field(min_length=1, max_length=100_000)


class Approval(BaseModel):
    approved: bool


class MemoryItemBody(BaseModel):
    """One hand-written item: its value and the kind it joins the layer under."""

    value: str = Field(max_length=4000)
    kind: str = Field(min_length=1, max_length=40)


class UpdateMemory(BaseModel):
    """A hand-routed version of one layer: `key -> {value, kind}`.

    The kind travels with the item, not with the layer: one layer holds items
    of several kinds, so rewriting it must not relabel them all. An empty value
    drops the key.
    """

    items: dict[str, MemoryItemBody]


class WriteMemoryItem(BaseModel):
    """One item routed by hand straight from the transcript."""

    key: str = Field(min_length=1, max_length=120)
    value: str = Field(min_length=1, max_length=4000)
    kind: str = Field(min_length=1, max_length=40)
    seq: int | None = Field(default=None, ge=1)


class MoveMemoryItem(BaseModel):
    """Re-routing of one item; the kind is the target layer's own."""

    kind: str = Field(min_length=1, max_length=40)


class FinishTask(BaseModel):
    """The end of a task: which working items outlive it, and as what.

    `promote` is `key -> kind of the long-term layer`; everything the map does
    not name is what the task took with it.
    """

    promote: dict[str, str] = Field(default_factory=dict)


class DecideProposal(BaseModel):
    """A human verdict on one router candidate."""

    decision: Literal["accept", "reject"]


class CreateCheckpoint(BaseModel):
    seq: int = Field(ge=1)
    name: str | None = Field(default=None, max_length=200)


class PatchSession(BaseModel):
    user_profile_id: str | None


class TaskTransition(BaseModel):
    """One move of the task machine, as a human asks for it."""

    stage: Literal["planning", "execution", "validation", "done", "cancelled"]
    force: bool = False
    reason: str | None = Field(default=None, max_length=500)
    failed_criteria: list[str] = Field(default_factory=list, max_length=30)
    expected_revision: int | None = Field(default=None, ge=0)
    expected_invariant_hash: str | None = Field(default=None, min_length=64, max_length=64)


class TaskVersion(BaseModel):
    expected_revision: int | None = Field(default=None, ge=0)


class TaskStep(BaseModel):
    status: Literal["todo", "in_progress", "done", "blocked"] | None = None
    title: str | None = Field(default=None, max_length=200)
    note: str | None = Field(default=None, max_length=500)
    expected_revision: int | None = Field(default=None, ge=0)


class ForkSession(BaseModel):
    seq: int = Field(ge=1)
    title: str | None = Field(default=None, max_length=200)
    spec: AgentSpec | None = None


class UpdateInvariants(BaseModel):
    expected_version_hash: str = Field(min_length=64, max_length=64)
    rules: list[InvariantRule] = Field(default_factory=list, max_length=100)
    replan_task: bool = False


class HttpApprovals:
    """Approval port backed by a second HTTP call.

    The streaming generator parks on a future while the browser shows the diff;
    a timeout resolves to a refusal, because the one answer that must never be
    inferred from silence is yes.
    """

    def __init__(self, pending: dict[str, asyncio.Future], session_id: str) -> None:
        self._pending = pending
        self._session_id = session_id

    async def confirm(self, call: ToolCall, preview: str) -> bool:
        key = f"{self._session_id}:{call.id}"
        future: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        self._pending[key] = future
        try:
            return await asyncio.wait_for(future, APPROVAL_TIMEOUT_S)
        except asyncio.TimeoutError:
            return False
        finally:
            self._pending.pop(key, None)


class TaskBoundExecutor:
    """Refuse a tool that finishes waiting after its task changed or paused."""

    def __init__(self, executor, conn: sqlite3.Connection, task: dict) -> None:
        self._executor = executor
        self._conn = conn
        self._task_id = task["id"]
        self._revision = int(task["revision"])

    def describe(self):
        return self._executor.describe()

    def preview(self, call: ToolCall) -> str:
        return self._executor.preview(call)

    async def run(self, call: ToolCall) -> ToolResult:
        current = repo.get_task(self._conn, self._task_id)
        invariant_hash = None
        if current is not None:
            invariant_hash = self._conn.execute(
                "SELECT invariant_hash FROM sessions WHERE id = ?",
                (current["session_id"],),
            ).fetchone()["invariant_hash"]
        if (
            current is None
            or int(current["revision"]) != self._revision
            or current["paused"]
            or current["stage"] != "execution"
            or not current["approved_plan_hash"]
            or current["approved_invariant_hash"] != invariant_hash
        ):
            return ToolResult(
                call_id=call.id,
                output="Состояние задачи изменилось: действие отменено, обновите задачу",
                ok=False,
            )
        return await self._executor.run(call)


def _db(request: Request) -> sqlite3.Connection:
    return request.app.state.db


def _provider(request: Request, spec: AgentSpec):
    try:
        return request.app.state.providers.require(spec.model.provider)
    except LLMError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


def _llm_for(request: Request, spec: AgentSpec):
    provider = _provider(request, spec)
    assert provider.client is not None
    return provider.client


@schema_router.get("/agent-spec-schema")
def agent_spec_schema() -> dict:
    """The frontend renders this schema instead of owning a second form model."""
    schema = AgentSpec.model_json_schema()
    defaults = AgentSpec().model_dump(mode="json")
    for name, value in defaults.items():
        schema["properties"][name]["default"] = value
    return schema


@schema_router.get("/models")
async def models(request: Request) -> list[dict]:
    try:
        return await request.app.state.providers.list_models()
    except Exception as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


def _workspace(raw: str | None) -> Path | None:
    """A workspace has to exist before an agent is pointed at it."""
    if not raw:
        return None
    path = Path(raw).expanduser()
    if not path.is_dir():
        raise HTTPException(status_code=400, detail=f"Папка не найдена: {raw}")
    return path.resolve()


def _executor(request: Request, spec: AgentSpec):
    """Select an infrastructure adapter; the agent core stays executor-agnostic."""
    root = Path(spec.tools.workspace_path or "")
    if spec.tools.executor == "docker":
        settings = request.app.state.settings
        return DockerExecutor(
            root,
            image=settings.docker_image,
            limits=DockerLimits(
                cpus=settings.docker_cpus,
                memory_mb=settings.docker_memory_mb,
                pids=settings.docker_pids_limit,
            ),
            timeout=spec.tools.command_timeout_s,
        )
    return LocalExecutor(root, timeout=spec.tools.command_timeout_s)


def _validated_tools(spec: AgentSpec) -> AgentSpec:
    workspace = _workspace(spec.tools.workspace_path)
    if workspace is None or str(workspace) == spec.tools.workspace_path:
        return spec
    return spec.model_copy(
        update={"tools": spec.tools.model_copy(update={"workspace_path": str(workspace)})}
    )


def _legacy_session_tools(spec: AgentSpec, body: CreateSession) -> AgentSpec:
    """Keep old clients readable while persistence is owned only by AgentSpec v2."""
    if body.workspace_path is None and body.executor_kind is None:
        return _validated_tools(spec)
    workspace = _workspace(body.workspace_path)
    tools = spec.tools.model_copy(
        update={
            "workspace_path": str(workspace) if workspace else None,
            "executor": body.executor_kind or spec.tools.executor,
        }
    )
    return spec.model_copy(update={"tools": tools})


def _load_agent(
    request: Request,
    session_id: str,
    model_context_window: int | None = None,
    invariants: InvariantSet | None = None,
    task_snapshot: dict | None = None,
) -> Agent:
    conn = _db(request)
    session = repo.get_session(conn, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Сессия не найдена")
    spec = repo.get_spec(conn, session["spec_hash"])
    if spec is None:  # pragma: no cover - foreign key makes this unreachable
        raise HTTPException(status_code=500, detail="Конфиг сессии потерян")

    task_row = task_snapshot if task_snapshot is not None else repo.active_task(conn, session_id)
    task = TaskState.from_dict(task_row) if task_row is not None else None
    # The stage decides the tools, so an executor is not even created where
    # the agent may not act: no text in the UI can widen that.
    acting = task is None or (not task.paused and task.stage != "planning")
    executor = None
    if spec.tools.workspace_path and acting:
        try:
            executor = request.app.state.executor_pool.wrap(_executor(request, spec))
            if task_row is not None:
                executor = TaskBoundExecutor(executor, conn, task_row)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    provider = _provider(request, spec)
    assert provider.client is not None
    agent = Agent(
        spec,
        request.app.state.provider_gate.wrap(provider.client),
        executor=executor,
        approvals=HttpApprovals(request.app.state.approvals, session_id),
        task=task,
        model_context_window=model_context_window,
        chars_per_token=session["chars_per_token"],
        message_format=provider.capabilities.transport_for(spec.model.model),
        user_profile=_session_user_profile(conn, session),
        invariants=invariants or _session_invariant_set(conn, session_id),
    )
    history = repo.load_messages(conn, session_id)
    summary = repo.latest_summary(conn, session_id)
    if summary is not None:
        agent.load_summary(summary)
    working, longterm = _memory_layers(conn, session, spec)
    agent.load_memory(working=working, longterm=longterm)
    # Planning must not even replay earlier tool exchanges to the provider.
    # The full transcript remains in SQLite and is restored again on execution.
    planning = task is not None and task.stage == "planning"
    agent.load_history([entry for entry in history if not planning or entry["role"] != "tool"])

    # Spend is cumulative across restarts, so the budget keeps its meaning.
    totals = repo.session_totals(conn, session_id)
    agent.ledger.requests = totals["requests"]
    agent.ledger.tokens_in = totals["tokens_in"]
    agent.ledger.tokens_out = totals["tokens_out"]
    agent.ledger.reasoning = totals["reasoning"]
    agent.ledger.cost_usd = totals["cost_usd"]
    return agent


@router.post("")
def create_session(body: CreateSession, request: Request) -> dict:
    preset = None
    if body.preset_id is not None:
        preset = repo.get_preset(_db(request), body.preset_id)
        if preset is None:
            raise HTTPException(status_code=404, detail="Пресет не найден")
    elif body.spec is None:
        preset = repo.get_default_preset(_db(request))
    spec = _legacy_session_tools(
        body.spec or (AgentSpec(**preset["spec"]) if preset else AgentSpec()), body
    )
    if "user_profile_id" not in body.model_fields_set:
        user_profile = repo.get_default_user_profile(_db(request))
    elif body.user_profile_id is None:
        user_profile = None
    else:
        user_profile = _user_profile(request, body.user_profile_id)
    user_profile_id = user_profile["id"] if user_profile else None
    session_id = repo.create_session(_db(request), spec, body.title, user_profile_id)
    return {
        "id": session_id,
        "spec": spec.model_dump(),
        "preset_id": preset["id"] if preset else None,
        "user_profile_id": user_profile_id,
        "invariant_hash": repo.session_invariants(_db(request), session_id)["version_hash"],
        "workspace_path": spec.tools.workspace_path,
        "executor_kind": spec.tools.executor,
    }


def _session_user_profile(
    conn: sqlite3.Connection, session: sqlite3.Row
) -> UserProfile | None:
    """The version the session's pointer names right now.

    Read on every call rather than cached: an edit to the profile applies from
    the next request in every session that points at it.
    """
    if not session["user_profile_id"]:
        return None
    pointer = repo.get_user_profile(conn, session["user_profile_id"])
    return UserProfile.model_validate(pointer["profile"]) if pointer else None


def _session_invariant_set(conn: sqlite3.Connection, session_id: str) -> InvariantSet:
    stored = repo.session_invariants(conn, session_id)
    return InvariantSet(rules=stored["rules"] if stored else [])


def _user_profile(request: Request, profile_id: str) -> dict:
    profile = repo.get_user_profile(_db(request), profile_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="Профиль пользователя не найден")
    return profile


@router.get("")
def list_sessions(request: Request) -> list[dict]:
    return repo.list_sessions(_db(request))


@router.get("/{session_id}")
def get_session(session_id: str, request: Request) -> dict:
    conn = _db(request)
    session = repo.get_session(conn, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Сессия не найдена")
    spec = repo.get_spec(conn, session["spec_hash"])
    messages = repo.load_messages(conn, session_id)
    summary = repo.latest_summary(conn, session_id)
    working, longterm = _memory_layers(conn, session, spec)
    role = spec.model.role if spec else ""
    user_profile = _session_user_profile(conn, session)
    task = _task_state(conn, session_id)
    invariants = _session_invariant_set(conn, session_id)
    suffix = stage_role(task)
    if spec and suffix:
        role = f"{role}\n\n{suffix}"
    context = (
        build_context_view(
            spec,
            messages,
            role=role,
            chars_per_token=session["chars_per_token"],
            summary=summary,
            working=working,
            longterm=longterm,
            invariants=invariants,
            user_profile=user_profile,
            task=task,
            limit_tokens=spec.context.window_tokens,
        ).as_dict(
            limit_tokens=spec.context.window_tokens,
            limit_source="spec" if spec.context.window_tokens is not None else None,
            overflow=spec.context.overflow,
        )
        if spec
        else None
    )
    return {
        "id": session["id"],
        "title": session["title"],
        "created_at": session["created_at"],
        "editable": session["run_id"] is None,
        "spec_hash": session["spec_hash"],
        "workspace_path": spec.tools.workspace_path if spec else None,
        "executor_kind": spec.tools.executor if spec else "local",
        "task": _task_payload(conn, session_id),
        # The journal covers closed tasks too: a transcript mark must survive
        # the `done` that removed the task from the strip.
        "task_journal": repo.session_task_transitions(conn, session_id),
        "spec": spec.model_dump() if spec else None,
        "user_profile_id": session["user_profile_id"],
        "user_profile": (
            repo.get_user_profile(conn, session["user_profile_id"])
            if session["user_profile_id"]
            else None
        ),
        "invariants": repo.session_invariants(conn, session_id),
        "invariant_history": repo.invariant_history(conn, session_id),
        "invariant_checks": repo.list_invariant_checks(conn, session_id),
        "messages": messages,
        "context": context,
        "totals": repo.session_totals(conn, session_id),
        "usage": repo.session_usage(conn, session_id),
        "summaries": repo.list_summaries(conn, session_id),
        "memory": _memory_detail(conn, session, spec),
        "checkpoints": repo.list_checkpoints(conn, session_id),
        "parent_id": session["parent_id"],
        "fork_seq": session["fork_seq"],
    }


@router.post("/{session_id}/checkpoints")
def create_checkpoint(
    session_id: str, body: CreateCheckpoint, request: Request
) -> dict:
    conn = _db(request)
    if repo.get_session(conn, session_id) is None:
        raise HTTPException(status_code=404, detail="Сессия не найдена")
    name = (body.name or "").strip() or f"checkpoint #{body.seq}"
    if not repo.save_checkpoint(conn, session_id, body.seq, name):
        raise HTTPException(status_code=400, detail=f"Сообщение #{body.seq} не найдено")
    return get_session(session_id, request)


@router.post("/{session_id}/fork")
def fork_session(session_id: str, body: ForkSession, request: Request) -> dict:
    conn = _db(request)
    if repo.get_session(conn, session_id) is None:
        raise HTTPException(status_code=404, detail="Сессия не найдена")
    title = (body.title or "").strip() or None
    spec = _validated_tools(body.spec) if body.spec is not None else None
    fork_id = repo.fork_session(conn, session_id, body.seq, title, spec=spec)
    if fork_id is None:
        raise HTTPException(status_code=400, detail=f"Сообщение #{body.seq} не найдено")
    if spec is not None and spec.memory.longterm_enabled:
        repo.attach_scope(conn, fork_id, spec.memory.longterm_scope)
    return get_session(fork_id, request)


@router.patch("/{session_id}")
def patch_session(session_id: str, body: PatchSession, request: Request) -> dict:
    """Switch the person the session talks to; applies from the next call."""
    conn = _db(request)
    session = repo.get_session(conn, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Сессия не найдена")
    if session["run_id"] is not None:
        raise HTTPException(
            status_code=409, detail="Сессия матричного прогона неизменяема"
        )
    if body.user_profile_id is not None:
        _user_profile(request, body.user_profile_id)
    repo.set_session_user_profile(conn, session_id, body.user_profile_id)
    return get_session(session_id, request)


@router.patch("/{session_id}/spec")
async def patch_session_spec(session_id: str, spec: AgentSpec, request: Request) -> dict:
    conn = _db(request)
    session = repo.get_session(conn, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Сессия не найдена")
    if session["run_id"] is not None:
        raise HTTPException(
            status_code=409, detail="Конфигурация сессии матричного прогона неизменяема"
        )
    spec = _validated_tools(spec)
    current = repo.get_spec(conn, session["spec_hash"])
    history = repo.load_messages(conn, session_id)
    extracted = None
    if (
        current is not None
        and not current.memory.working_enabled
        and spec.memory.working_enabled
        and history
    ):
        try:
            extracted = await _route_memory(
                request, session, spec, history, from_start=True
            )
        except (BudgetExceeded, LLMError) as error:
            raise HTTPException(status_code=502, detail=str(error)) from error
        if extracted is not None and extracted.error is not None:
            repo.save_spec(conn, spec)
            _record_memory_usage(conn, session_id, spec, extracted)
            raise HTTPException(status_code=502, detail=extracted.error)
    result = repo.update_session_spec(conn, session_id, spec)
    if result != "updated":  # pragma: no cover - guarded in this request transaction
        raise HTTPException(status_code=409, detail="Сессия изменилась во время обновления")
    if spec.memory.longterm_enabled:
        repo.attach_scope(conn, session_id, spec.memory.longterm_scope)
    if extracted is not None:
        _apply_memory(conn, repo.get_session(conn, session_id), spec, extracted)
    return get_session(session_id, request)


@router.put("/{session_id}/invariants")
def update_session_invariants(
    session_id: str, body: UpdateInvariants, request: Request
) -> dict:
    """Replace the complete set only from the version the editor actually saw."""
    conn = _db(request)
    try:
        result = repo.set_session_invariants(
            conn,
            session_id,
            InvariantSet(rules=body.rules),
            expected_hash=body.expected_version_hash,
            replan_task=body.replan_task,
        )
    except TaskError as error:
        raise HTTPException(status_code=409, detail=error.message) from error
    except sqlite3.IntegrityError as error:
        if str(error) != "invariant version changed concurrently":
            raise
        result = "conflict"
    if result == "missing":
        raise HTTPException(status_code=404, detail="Сессия не найдена")
    if result == "run":
        raise HTTPException(status_code=409, detail="Сессия матричного прогона неизменяема")
    if result == "conflict":
        raise HTTPException(
            status_code=409,
            detail="Набор инвариантов уже изменён. Обновите редактор и повторите правку.",
        )
    if result == "replan":
        raise HTTPException(
            status_code=409,
            detail="Активная задача требует явного возврата в планирование.",
        )
    return get_session(session_id, request)


@router.put("/{session_id}/memory/{layer}")
def update_session_memory(
    session_id: str, layer: str, body: UpdateMemory, request: Request
) -> dict:
    """Rewrite one layer by hand; the version chain records who routed it."""
    conn, session, spec, owner = _memory_target(request, session_id, layer)
    previous = repo.latest_memory(conn, layer, **owner)
    stored = dict((previous or {}).get("items") or {})
    seq = _last_seq(conn, session_id, previous)
    items: dict[str, dict] = {}
    for key, item in body.items.items():
        if not item.value:
            continue
        _check_kind(layer, item.kind)
        before = stored.get(key)
        # An item the hand left alone keeps the route that produced it: only
        # what actually changed here becomes manual.
        if before and before.get("value") == item.value and before.get("kind") == item.kind:
            items[key] = before
        else:
            items[key] = {
                "value": item.value,
                "kind": item.kind,
                "origin": "manual",
                "seq": seq,
            }
    _write_layer(conn, layer, owner, items=items, covered_seq=seq, origin="manual")
    return get_session(session_id, request)


@router.post("/{session_id}/memory/{layer}/items")
def write_memory_item(
    session_id: str, layer: str, body: WriteMemoryItem, request: Request
) -> dict:
    """Remember one thing from the transcript into a named layer and kind.

    The point write exists so «запомнить» on a message does not have to resend
    the whole layer: a stale copy in the browser would quietly resurrect items
    the router dropped in the meantime.
    """
    conn, session, spec, owner = _memory_target(request, session_id, layer)
    _check_kind(layer, body.kind)
    previous = repo.latest_memory(conn, layer, **owner)
    items = dict((previous or {}).get("items") or {})
    covered_seq = _last_seq(conn, session_id, previous)
    items[body.key] = {
        "value": body.value,
        "kind": body.kind,
        "origin": "manual",
        # The item points at the message it came from, not at the last one.
        "seq": body.seq if body.seq is not None else covered_seq,
    }
    _write_layer(conn, layer, owner, items=items, covered_seq=covered_seq, origin="manual")
    return get_session(session_id, request)


@router.delete("/{session_id}/memory/{layer}/items/{key}")
def forget_memory_item(session_id: str, layer: str, key: str, request: Request) -> dict:
    """Forget one item; every version the layer ever had stays readable."""
    conn, session, spec, owner = _memory_target(request, session_id, layer)
    previous = repo.latest_memory(conn, layer, **owner)
    items = dict((previous or {}).get("items") or {})
    if items.pop(key, None) is None:
        raise HTTPException(status_code=404, detail=f"В слое {layer} нет ключа {key}")
    covered_seq = _last_seq(conn, session_id, previous)
    _write_layer(conn, layer, owner, items=items, covered_seq=covered_seq, origin="manual")
    return get_session(session_id, request)


@router.post("/{session_id}/memory/{layer}/items/{key}/move")
def move_memory_item(
    session_id: str, layer: str, key: str, body: MoveMemoryItem, request: Request
) -> dict:
    """Re-route one item to the other layer.

    Two versions, not one: the item leaves its source and joins its target, and
    each layer keeps the move in its own history. The item carries the seq it
    was routed from — where a fact came from does not change because the fact
    changed layers — and `origin='promote'`, which is neither the router nor a
    fresh hand-written value.
    """
    target_layer = "longterm" if layer == "working" else "working"
    conn, session, spec, source_owner = _memory_target(request, session_id, layer)
    target_owner = _layer_owner(conn, session, spec, target_layer)
    if target_owner is None:
        raise HTTPException(
            status_code=409, detail=f"Слой {target_layer} выключен в настройках"
        )
    _check_kind(target_layer, body.kind)
    source = repo.latest_memory(conn, layer, **source_owner)
    source_items = dict((source or {}).get("items") or {})
    item = source_items.pop(key, None)
    if item is None:
        raise HTTPException(status_code=404, detail=f"В слое {layer} нет ключа {key}")
    target = repo.latest_memory(conn, target_layer, **target_owner)
    target_items = dict((target or {}).get("items") or {})
    target_items[key] = {
        "value": item.get("value", ""),
        "kind": body.kind,
        "origin": "promote",
        "seq": item.get("seq", 0),
    }
    # A move reads no transcript, so neither watermark advances.
    _write_layer(
        conn,
        layer,
        source_owner,
        items=source_items,
        covered_seq=int((source or {}).get("covered_seq") or 0),
        origin="promote",
    )
    _write_layer(
        conn,
        target_layer,
        target_owner,
        items=target_items,
        covered_seq=int((target or {}).get("covered_seq") or 0),
        origin="promote",
    )
    return get_session(session_id, request)


def _close_working_layer(
    conn: sqlite3.Connection, session_id: str, request: Request
) -> None:
    """A closed task empties its working layer; a disabled layer is no error.

    This is the automatic half of `finish_task`: nothing is promoted, because
    nobody was asked. What deserves to outlive the task is promoted by hand
    before closing it, or learned again by the router.
    """
    session = repo.get_session(conn, session_id)
    if session is None or session["run_id"] is not None:
        return
    spec = repo.get_spec(conn, session["spec_hash"])
    owner = _layer_owner(conn, session, spec, "working")
    if owner is None:
        return
    previous = repo.latest_memory(conn, "working", **owner)
    if previous is None or not (previous.get("items") or {}):
        return
    _write_layer(
        conn,
        "working",
        owner,
        items={},
        covered_seq=_last_seq(conn, session_id, previous),
        origin="manual",
    )


@router.post("/{session_id}/memory/working/finish")
def finish_task(session_id: str, body: FinishTask, request: Request) -> dict:
    """Close the task: keep what was chosen, empty the working layer.

    Emptying is a new version, not a deletion — the finished task stays
    readable in the version chain. The router watermark is left where it is:
    it has already read this dialogue, so the details of the closed task do not
    come back on the next turn while the long-term layer keeps what was
    promoted out of them.
    """
    conn, session, spec, owner = _memory_target(request, session_id, "working")
    previous = repo.latest_memory(conn, "working", **owner)
    items = dict((previous or {}).get("items") or {})
    promoted: dict[str, dict] = {}
    for key, kind in body.promote.items():
        item = items.get(key)
        if item is None:
            raise HTTPException(status_code=404, detail=f"В слое working нет ключа {key}")
        _check_kind("longterm", kind)
        promoted[key] = {
            "value": item.get("value", ""),
            "kind": kind,
            "origin": "promote",
            "seq": item.get("seq", 0),
        }
    if promoted:
        target_owner = _layer_owner(conn, session, spec, "longterm")
        if target_owner is None:
            raise HTTPException(
                status_code=409, detail="Слой longterm выключен в настройках"
            )
        target = repo.latest_memory(conn, "longterm", **target_owner)
        target_items = dict((target or {}).get("items") or {})
        target_items.update(promoted)
        _write_layer(
            conn,
            "longterm",
            target_owner,
            items=target_items,
            covered_seq=int((target or {}).get("covered_seq") or 0),
            origin="promote",
        )
    _write_layer(
        conn,
        "working",
        owner,
        items={},
        covered_seq=_last_seq(conn, session_id, previous),
        origin="promote" if promoted else "manual",
    )
    return get_session(session_id, request)


@router.post("/{session_id}/memory/proposals/{proposal_id}")
def decide_proposal(
    session_id: str, proposal_id: int, body: DecideProposal, request: Request
) -> dict:
    """Accept or reject one router candidate.

    An accepted item keeps `origin='router'` — the router is who produced it —
    while the version it lands in is `manual`, because a human is who wrote it.
    """
    conn = _db(request)
    session = repo.get_session(conn, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Сессия не найдена")
    proposal = repo.get_proposal(conn, proposal_id)
    spec = repo.get_spec(conn, session["spec_hash"])
    owner = (
        _layer_owner(conn, session, spec, proposal["layer"]) if proposal else None
    )
    # A candidate is addressed through its session: the id alone would let one
    # session decide another session's memory, or a scope it never joined.
    if proposal is None or owner is None or not _owns(proposal, owner):
        raise HTTPException(status_code=404, detail="Кандидат не найден")
    if body.decision == "reject":
        repo.delete_proposal(conn, proposal_id)
        return get_session(session_id, request)
    if proposal["layer"] == "profile":
        _accept_profile_proposal(conn, proposal)
        return get_session(session_id, request)
    previous = repo.latest_memory(conn, proposal["layer"], **owner)
    items = dict((previous or {}).get("items") or {})
    if proposal["value"]:
        items[proposal["key"]] = {
            "value": proposal["value"],
            "kind": proposal["kind"],
            "origin": "router",
            "seq": proposal["source_seq"],
        }
    else:
        items.pop(proposal["key"], None)
    repo.save_memory(
        conn,
        proposal["layer"],
        **owner,
        covered_seq=max(
            int(proposal["source_seq"]), int((previous or {}).get("covered_seq") or 0)
        ),
        items=items,
        tokens=_items_tokens(items),
        origin="manual",
    )
    repo.delete_proposal(conn, proposal_id)
    return get_session(session_id, request)


def _accept_profile_proposal(conn: sqlite3.Connection, proposal: dict) -> None:
    """Write one field into the profile the candidate belongs to.

    The value was valid when routed, but the profile may have been edited
    since, so the whole profile is checked again before it becomes a version.
    """
    pointer = repo.get_user_profile(conn, proposal["user_profile_id"])
    if pointer is None:  # pragma: no cover - cascade removes the candidate
        raise HTTPException(status_code=404, detail="Профиль пользователя не найден")
    try:
        profile = UserProfile.model_validate(
            {**pointer["profile"], proposal["key"]: json.loads(proposal["value"])}
        )
    except (ValidationError, json.JSONDecodeError) as error:
        raise HTTPException(
            status_code=422, detail=f"Кандидат не проходит схему профиля: {error}"
        ) from error
    repo.update_user_profile(conn, pointer["id"], profile)
    repo.delete_proposal(conn, proposal["id"])


def _task_state(conn: sqlite3.Connection, session_id: str) -> TaskState | None:
    """The active task of the session as the machine sees it."""
    task = repo.active_task(conn, session_id)
    return TaskState.from_dict(task) if task is not None else None


def _task_payload(conn: sqlite3.Connection, session_id: str) -> dict | None:
    """Everything the strip needs: state, journal and the buttons to draw.

    The allowed transitions are computed here, so the client never keeps its
    own copy of the transition table.
    """
    task = repo.active_task(conn, session_id)
    if task is None:
        return None
    state = TaskState.from_dict(task)
    return {
        **task,
        "transitions": repo.list_task_transitions(conn, task["id"]),
        "available": [item.to_dict() for item in available_transitions(state)],
    }


def _require_task(conn: sqlite3.Connection, session_id: str) -> dict:
    if repo.get_session(conn, session_id) is None:
        raise HTTPException(status_code=404, detail="Сессия не найдена")
    task = repo.active_task(conn, session_id)
    if task is None:
        raise HTTPException(status_code=409, detail="У сессии нет активной задачи")
    return task


def _reject_task(conn: sqlite3.Connection, task: dict, error: TaskError) -> HTTPException:
    """Persist and return one explainable refusal with the allowed next action."""
    try:
        repo.record_task_rejection(
            conn,
            task["id"],
            actor="user",
            reason=f"{error.code}: {error.message}",
        )
    except TaskError:
        pass
    current = repo.get_task(conn, task["id"])
    action = (
        TaskState.from_dict(current).to_dict()["expected_action"]
        if current is not None
        else {"actor": "user", "text": "обновить состояние задачи"}
    )
    return HTTPException(
        status_code=409,
        detail={"code": error.code, "message": error.message, "next_action": action},
    )


@router.get("/{session_id}/task")
def get_task(session_id: str, request: Request) -> dict:
    conn = _db(request)
    if repo.get_session(conn, session_id) is None:
        raise HTTPException(status_code=404, detail="Сессия не найдена")
    payload = _task_payload(conn, session_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="У сессии нет активной задачи")
    return payload


@router.post("/{session_id}/task")
def start_task(session_id: str, request: Request) -> dict:
    conn = _db(request)
    if repo.get_session(conn, session_id) is None:
        raise HTTPException(status_code=404, detail="Сессия не найдена")
    if repo.create_task(conn, session_id) is None:
        raise HTTPException(status_code=409, detail="У сессии уже есть активная задача")
    return get_session(session_id, request)


@router.post("/{session_id}/task/transition")
def move_task(session_id: str, body: TaskTransition, request: Request) -> dict:
    conn = _db(request)
    task = _require_task(conn, session_id)
    try:
        if body.force:
            raise TaskError("force_removed", "Принудительные переходы отключены")
        if body.expected_revision is None or body.expected_invariant_hash is None:
            raise TaskError(
                "version_required",
                "Переход требует текущих версий задачи и инвариантов; обновите экран",
            )
        moved = repo.apply_transition(
            conn,
            task["id"],
            body.stage,
            force=body.force,
            failed_criteria=tuple(body.failed_criteria),
            reason=body.reason,
            expected_revision=body.expected_revision,
            expected_invariant_hash=body.expected_invariant_hash,
        )
    except TaskError as error:
        raise _reject_task(conn, task, error) from error
    if moved["stage"] in ("done", "cancelled"):
        # Closing the task closes the working layer with it: the finished task
        # keeps one owner from its first step to its last.
        _close_working_layer(conn, session_id, request)
    return get_session(session_id, request)


@router.post("/{session_id}/task/pause")
def pause_task(session_id: str, body: TaskVersion, request: Request) -> dict:
    return _set_pause(session_id, True, body, request)


@router.post("/{session_id}/task/resume")
def resume_task(session_id: str, body: TaskVersion, request: Request) -> dict:
    return _set_pause(session_id, False, body, request)


def _set_pause(
    session_id: str, paused: bool, body: TaskVersion, request: Request
) -> dict:
    conn = _db(request)
    task = _require_task(conn, session_id)
    try:
        if body.expected_revision is None:
            raise TaskError("version_required", "Обновите состояние задачи перед действием")
        repo.set_task_paused(
            conn, task["id"], paused, expected_revision=body.expected_revision
        )
    except TaskError as error:
        raise _reject_task(conn, task, error) from error
    if not paused:
        # The tracker intentionally sleeps while a task is paused. Skip that
        # discussion on resume instead of treating unrelated paused turns as
        # evidence that the current step or validation result changed.
        repo.set_task_covered(conn, task["id"], _last_seq(conn, session_id, None))
    return get_session(session_id, request)


@router.patch("/{session_id}/task/steps/{index}")
def edit_task_step(
    session_id: str, index: int, body: TaskStep, request: Request
) -> dict:
    conn = _db(request)
    task = _require_task(conn, session_id)
    try:
        if body.expected_revision is None:
            raise TaskError("version_required", "Обновите состояние задачи перед правкой")
        repo.update_task_step(
            conn,
            task["id"],
            index,
            status=body.status,
            title=body.title,
            note=body.note,
            expected_revision=body.expected_revision,
        )
    except TaskError as error:
        code = 404 if error.code == "step_index" else 409
        if code == 404:
            raise HTTPException(status_code=code, detail=error.message) from error
        raise _reject_task(conn, task, error) from error
    return get_session(session_id, request)


@router.delete("/{session_id}")
def delete_session(session_id: str, request: Request) -> dict:
    if not repo.delete_session(_db(request), session_id):
        raise HTTPException(status_code=404, detail="Сессия не найдена")
    return {"deleted": session_id}


@router.post("/{session_id}/approvals/{call_id}")
async def resolve_approval(
    session_id: str, call_id: str, body: Approval, request: Request
) -> dict:
    """Answer a pending confirmation. Async on purpose: the future lives in this loop."""
    future = request.app.state.approvals.get(f"{session_id}:{call_id}")
    if future is None or future.done():
        raise HTTPException(status_code=404, detail="Подтверждение уже не ожидается")
    future.set_result(body.approved)
    return {"call_id": call_id, "approved": body.approved}


@router.post("/{session_id}/messages")
async def send_message(
    session_id: str, body: SendMessage, request: Request
) -> StreamingResponse:
    conn = _db(request)
    session = repo.get_session(conn, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Сессия не найдена")
    spec = repo.get_spec(conn, session["spec_hash"])
    if spec is None:  # pragma: no cover - foreign key makes this unreachable
        raise HTTPException(status_code=500, detail="Конфиг сессии потерян")
    model_context_window = await _model_context_window(request, spec)
    llm = request.app.state.provider_gate.wrap(_llm_for(request, spec))
    history = repo.load_messages(conn, session_id)
    previous_summary = repo.latest_summary(conn, session_id)
    # One turn owns one immutable rules version even if another request edits
    # the session while a helper call is still running.
    invariant_hash = session["invariant_hash"]
    invariant_set = _session_invariant_set(conn, session_id)
    turn_task = repo.active_task(conn, session_id)

    async def events() -> AsyncIterator[str]:
        totals = repo.session_totals(conn, session_id)
        ledger = Ledger(
            budget_usd=spec.context.budget_usd,
            requests=totals["requests"],
            tokens_in=totals["tokens_in"],
            tokens_out=totals["tokens_out"],
            reasoning=totals["reasoning"],
            cost_usd=totals["cost_usd"],
        )
        strategy = strategy_for(spec.context)
        try:
            compressed = (
                await strategy.summarize_if_due(llm, history, previous_summary, ledger)
                if isinstance(strategy, SummaryContextStrategy)
                else None
        )
        except (BudgetExceeded, LLMError) as error:
            payload = {"type": "error", "message": str(error)}
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            return
        if compressed is not None:
            stored = repo.save_summary(
                conn,
                session_id,
                covered_seq=compressed.covered_seq,
                message_count=compressed.message_count,
                text=compressed.text,
                tokens=compressed.tokens,
            )
            repo.record_usage(
                conn,
                session_id,
                spec.context.summary_model,
                compressed.usage,
                compressed.latency_ms,
                spec_hash=spec.hash,
                estimate={"input_tokens": compressed.input_estimate_tokens},
                context={
                    "estimated_tokens": compressed.input_estimate_tokens,
                    "input_tokens": compressed.usage.tokens_in,
                    "output_tokens": compressed.usage.tokens_out,
                },
                kind="summary",
                invariant_hash=invariant_hash,
            )
            yield f"data: {json.dumps({'type': 'summary', **stored}, ensure_ascii=False)}\n\n"

        agent = _load_agent(
            request,
            session_id,
            model_context_window,
            invariants=invariant_set,
            task_snapshot=turn_task,
        )
        repo.append_message(conn, session_id, "user", body.text)
        started: dict[str, ToolCall] = {}
        pending_usage: UsageReported | None = None
        async for event in agent.stream(body.text):
            if isinstance(event, ToolStarted):
                started[event.call_id] = ToolCall(
                    id=event.call_id, tool=event.tool, arguments_json=event.arguments
                )
            elif isinstance(event, ToolFinished):
                _store_tool_call(conn, session_id, event, started)
            elif isinstance(event, UsageReported):
                pending_usage = event
                continue
            elif isinstance(event, InvariantChecked):
                repo.record_invariant_check(
                    conn,
                    session_id,
                    invariant_hash,
                    event.target,
                    event.outcome,
                    event.rule_ids,
                    event.explanation,
                    event.alternative,
                    _last_seq(conn, session_id, None),
                )
                _record_invariant_usage(conn, session_id, spec, event, invariant_hash)
            elif isinstance(event, Done):
                # A turn that ended in tool calls alone leaves no text to store.
                message_id = None
                if event.text:
                    message_id = repo.append_message(conn, session_id, "assistant", event.text)
                if pending_usage is not None:
                    _record_turn_usage(
                        conn, session_id, agent, pending_usage, message_id, invariant_hash
                    )
                    yield _sse(pending_usage)
                    pending_usage = None
                session = repo.get_session(conn, session_id)
                task = repo.active_task(conn, session_id)
                task_unchanged = bool(
                    turn_task is not None
                    and task is not None
                    and task["id"] == turn_task["id"]
                    and task["revision"] == turn_task["revision"]
                )
                if (
                    task_unchanged
                    and task is not None
                    and task["stage"] == "planning"
                    and event.text
                ):
                    try:
                        repo.save_task_plan(
                            conn,
                            task["id"],
                            event.text,
                            expected_revision=turn_task["revision"],
                        )
                    except TaskError as error:
                        yield _json_sse({"type": "task_rejected", "reasons": [error.message]})
                    else:
                        yield _json_sse(
                            {"type": "task", "task": _task_payload(conn, session_id)}
                        )
                elif task_unchanged and task is not None and spec.task.tracker_enabled:
                    # The tracker and the memory router are two independent
                    # calls: one refusing to answer must not cost the other its
                    # turn, so each has its own error frame.
                    try:
                        marked = await _run_tracker(
                            request,
                            session_id,
                            spec,
                            turn_task,
                            repo.load_messages(conn, session_id),
                            invariant_set,
                        )
                    except (BudgetExceeded, LLMError) as error:
                        yield _json_sse({"type": "task_error", "message": str(error)})
                    else:
                        if marked is not None and marked.error is not None:
                            _record_tracker_usage(
                                conn, session_id, spec, marked, invariant_hash
                            )
                            yield _json_sse(
                                {"type": "task_error", "message": marked.error}
                            )
                        elif marked is not None:
                            for task_event in _apply_tracker(
                                conn, session_id, spec, turn_task, marked, invariant_hash
                            ):
                                yield _json_sse(task_event)
                elif turn_task is not None and not task_unchanged:
                    yield _json_sse(
                        {
                            "type": "task_rejected",
                            "reasons": [
                                "Состояние задачи изменилось во время ответа; поздний результат не применён"
                            ],
                        }
                    )
                if _router_layers(spec) and event.text and session is not None:
                    try:
                        extracted = await _route_memory(
                            request, session, spec, repo.load_messages(conn, session_id)
                        )
                    except (BudgetExceeded, LLMError) as error:
                        yield _json_sse({"type": "memory_error", "message": str(error)})
                    else:
                        if extracted is not None and extracted.error is not None:
                            _record_memory_usage(
                                conn, session_id, spec, extracted, invariant_hash
                            )
                            yield _json_sse(
                                {"type": "memory_error", "message": extracted.error}
                            )
                        elif extracted is not None:
                            for memory_event in _apply_memory(
                                conn, session, spec, extracted, invariant_hash
                            ):
                                yield _json_sse(memory_event)
            elif isinstance(event, Failed):
                if pending_usage is not None:
                    _record_turn_usage(
                        conn, session_id, agent, pending_usage, None, invariant_hash
                    )
                    yield _sse(pending_usage)
                    pending_usage = None
                repo.drop_last_user_message(conn, session_id)
            payload = dataclasses.asdict(event)
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _scope_id(conn: sqlite3.Connection, session: sqlite3.Row, spec: AgentSpec) -> str:
    """The long-term layer is addressed by scope; joining it is idempotent."""
    return repo.attach_scope(conn, session["id"], spec.memory.longterm_scope)


def _layer_owner(
    conn: sqlite3.Connection,
    session: sqlite3.Row,
    spec: AgentSpec | None,
    layer: str,
) -> dict | None:
    """Where a layer's versions live, or None when the spec keeps it off."""
    if spec is None:
        return None
    if layer == "profile":
        profile_id = session["user_profile_id"]
        return {"user_profile_id": profile_id} if profile_id else None
    if layer == "working":
        return {"session_id": session["id"]} if spec.memory.working_enabled else None
    return {"scope_id": _scope_id(conn, session, spec)} if spec.memory.longterm_enabled else None


def _memory_target(
    request: Request, session_id: str, layer: str
) -> tuple[sqlite3.Connection, sqlite3.Row, AgentSpec | None, dict]:
    """Resolve one writable layer of one editable session, or refuse.

    Every hand write goes through here, so a disabled layer, a matrix session
    and an invented layer name are refused once instead of in four endpoints.
    """
    if layer not in LAYER_KINDS:
        raise HTTPException(status_code=404, detail=f"Нет слоя памяти {layer}")
    conn = _db(request)
    session = repo.get_session(conn, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Сессия не найдена")
    if session["run_id"] is not None:
        raise HTTPException(status_code=409, detail="Память матричной сессии неизменяема")
    spec = repo.get_spec(conn, session["spec_hash"])
    owner = _layer_owner(conn, session, spec, layer)
    if owner is None:
        raise HTTPException(status_code=409, detail=f"Слой {layer} выключен в настройках")
    return conn, session, spec, owner


def _check_kind(layer: str, kind: str) -> None:
    if kind not in LAYER_KINDS[layer]:
        raise HTTPException(
            status_code=422,
            detail=f"Вид {kind} не из слоя {layer}: " + ", ".join(LAYER_KINDS[layer]),
        )


def _last_seq(
    conn: sqlite3.Connection, session_id: str, previous: dict | None
) -> int:
    """How far a hand-written version has seen the dialogue."""
    history = repo.load_messages(conn, session_id)
    if history:
        return int(history[-1]["seq"])
    return int((previous or {}).get("covered_seq") or 0)


def _write_layer(
    conn: sqlite3.Connection,
    layer: str,
    owner: dict,
    *,
    items: dict[str, dict],
    covered_seq: int,
    origin: str,
) -> None:
    repo.save_memory(
        conn,
        layer,
        **owner,
        covered_seq=covered_seq,
        items=items,
        tokens=_items_tokens(items),
        origin=origin,
    )


def _memory_layers(
    conn: sqlite3.Connection, session: sqlite3.Row, spec: AgentSpec | None
) -> tuple[dict | None, dict | None]:
    """Only an enabled layer reaches the payload — that is the routing switch."""
    layers = []
    for layer in ("working", "longterm"):
        owner = _layer_owner(conn, session, spec, layer)
        layers.append(repo.latest_memory(conn, layer, **owner) if owner else None)
    return layers[0], layers[1]


def _memory_detail(
    conn: sqlite3.Connection, session: sqlite3.Row, spec: AgentSpec | None
) -> dict:
    working_owner = _layer_owner(conn, session, spec, "working")
    longterm_owner = _layer_owner(conn, session, spec, "longterm")
    profile_owner = _layer_owner(conn, session, spec, "profile")
    return {
        # Learning the profile needs the router; without it the panel says so.
        "profile": {
            "enabled": profile_owner is not None,
            "router": bool(spec and _router_layers(spec)),
            "proposals": repo.list_proposals(conn, "profile", **profile_owner)
            if profile_owner
            else [],
        },
        "working": {
            "enabled": working_owner is not None,
            "versions": repo.list_memory(conn, "working", **working_owner)
            if working_owner
            else [],
            "proposals": repo.list_proposals(conn, "working", **working_owner)
            if working_owner
            else [],
        },
        "longterm": {
            "enabled": longterm_owner is not None,
            "scope": spec.memory.longterm_scope if spec else None,
            # Sessions sharing the space, so the card can show what the long-term
            # layer reaches beyond this dialogue.
            "sessions": repo.scope_sessions(conn, spec.memory.longterm_scope)
            if spec
            else 0,
            "write": spec.memory.longterm_write if spec else None,
            "versions": repo.list_memory(conn, "longterm", **longterm_owner)
            if longterm_owner
            else [],
            "proposals": repo.list_proposals(conn, "longterm", **longterm_owner)
            if longterm_owner
            else [],
        },
    }


def _router_layers(spec: AgentSpec) -> tuple[str, ...]:
    """Which layers the router is asked about.

    `longterm_write=manual` is not «route and discard»: the layer is filled by
    hand only, so paying for a routing decision nobody will read is waste.
    """
    layers = ["working"] if spec.memory.working_enabled else []
    if spec.memory.longterm_enabled and spec.memory.longterm_write != "manual":
        layers.append("longterm")
    return tuple(layers)


async def _route_memory(
    request: Request,
    session: sqlite3.Row,
    spec: AgentSpec,
    history: list[dict],
    *,
    from_start: bool = False,
) -> MemoryResult | None:
    conn = _db(request)
    session_id = session["id"]
    layers = _router_layers(spec)
    if not layers:
        return None
    previous = {
        layer: repo.latest_memory(conn, layer, **owner)
        for layer in layers
        if (owner := _layer_owner(conn, session, spec, layer)) is not None
    }
    user_profile = _session_user_profile(conn, session)
    # A catch-up over the whole history fills memory layers; old requests
    # about answer style are not replayed as new profile candidates.
    if user_profile is not None and not from_start:
        layers = (*layers, "profile")
    totals = repo.session_totals(conn, session_id)
    ledger = Ledger(
        budget_usd=spec.context.budget_usd,
        requests=totals["requests"],
        tokens_in=totals["tokens_in"],
        tokens_out=totals["tokens_out"],
        reasoning=totals["reasoning"],
        cost_usd=totals["cost_usd"],
    )
    router = MemoryRouter(spec.memory.model, spec.memory.max_tokens, layers)
    return await router.extract(
        request.app.state.provider_gate.wrap(_llm_for(request, spec)),
        history,
        previous,
        ledger,
        covered_seq=0 if from_start else int(session["memory_covered_seq"]),
        user_profile=user_profile,
    )


def _apply_memory(
    conn: sqlite3.Connection,
    session: sqlite3.Row,
    spec: AgentSpec,
    extracted: MemoryResult,
    invariant_hash: str | None = None,
) -> list[dict]:
    """Store what the router routed and say what happened to each layer.

    The working layer is this task's own data and is written at once; memory
    that outlives the dialogue goes through `longterm_write`, so under
    `confirm` the router only fills a candidate list. The profile is declared
    by the person, so its changes are always candidates, whatever the mode.
    """
    assert extracted.updates is not None
    session_id = session["id"]
    events: list[dict] = []
    for layer, update in extracted.updates.items():
        owner = _layer_owner(conn, session, spec, layer)
        if owner is None or not update.changes:
            continue
        if layer == "profile" or (
            layer == "longterm" and spec.memory.longterm_write == "confirm"
        ):
            proposals = repo.save_proposals(
                conn,
                layer,
                **owner,
                changes=update.changes,
                source_seq=extracted.covered_seq,
            )
            events.append({"type": "memory_proposals", "layer": layer, "proposals": proposals})
            continue
        stored = repo.save_memory(
            conn,
            layer,
            **owner,
            covered_seq=extracted.covered_seq,
            items=update.items,
            tokens=_items_tokens(update.items),
            origin="router",
        )
        events.append({"type": "memory", "layer": layer, **stored})
    if extracted.rejected:
        events.append({"type": "memory_rejected", "keys": extracted.rejected})
    repo.set_memory_covered(conn, session_id, extracted.covered_seq)
    _record_memory_usage(conn, session_id, spec, extracted, invariant_hash)
    return events


async def _run_tracker(
    request: Request,
    session_id: str,
    spec: AgentSpec,
    task: dict,
    history: list[dict],
    invariants: InvariantSet,
) -> TrackerResult | None:
    """Ask the tracker what the finished turn did to the task."""
    conn = _db(request)
    state = TaskState.from_dict(task)
    if not tracker_runs(state):
        return None
    totals = repo.session_totals(conn, session_id)
    ledger = Ledger(
        budget_usd=spec.context.budget_usd,
        requests=totals["requests"],
        tokens_in=totals["tokens_in"],
        tokens_out=totals["tokens_out"],
        reasoning=totals["reasoning"],
        cost_usd=totals["cost_usd"],
    )
    tracker = TaskTracker(spec.task.model, spec.task.max_tokens)
    return await tracker.observe(
        request.app.state.provider_gate.wrap(_llm_for(request, spec)),
        state,
        history,
        ledger,
        covered_seq=int(task["covered_seq"]),
        invariants=invariants,
    )


def _apply_tracker(
    conn: sqlite3.Connection,
    session_id: str,
    spec: AgentSpec,
    task: dict,
    marked: TrackerResult,
    invariant_hash: str | None = None,
) -> list[dict]:
    """Put the tracker's answer through the machine and say what survived.

    Nothing here trusts the answer: every part goes through the same guards a
    human's move goes through, and what they refuse comes back as a reason the
    strip can show.
    """
    task_id = task["id"]
    reasons = list(marked.rejected)
    proposed: str | None = None
    current = repo.get_task(conn, task_id)
    if (
        current is None
        or current["revision"] != task["revision"]
        or current["paused"]
        or current["stage"] != task["stage"]
    ):
        reasons.append(
            "Состояние задачи изменилось во время ответа; решение трекера не применено"
        )
        repo.record_task_rejection(
            conn,
            task_id,
            actor="tracker",
            reason="; ".join(reasons),
            seq=marked.covered_seq,
        )
        _record_tracker_usage(conn, session_id, spec, marked, invariant_hash)
        return [
            {"type": "task", "task": _task_payload(conn, session_id), "proposed": None},
            {"type": "task_rejected", "reasons": reasons},
        ]
    revision = int(current["revision"])
    if marked.step is not None and (marked.step.status or marked.step.note):
        try:
            changed = repo.update_task_step(
                conn,
                task_id,
                marked.step.index,
                status=marked.step.status,
                note=marked.step.note,
                actor="tracker",
                seq=marked.covered_seq,
                expected_revision=revision,
            )
            revision = int(changed["revision"])
        except TaskError as error:
            reasons.append(error.message)
    if marked.verdict is not None:
        try:
            changed = repo.set_task_verdict(
                conn,
                task_id,
                marked.verdict,
                actor="tracker",
                seq=marked.covered_seq,
                expected_revision=revision,
            )
            revision = int(changed["revision"])
        except TaskError as error:
            reasons.append(error.message)
    if marked.propose is not None:
        # Checked against the state the marks left behind: a step closed a
        # moment ago is what opens the move into validation.
        state = _task_state(conn, session_id)
        allowed = (
            {item.stage for item in available_transitions(state, "tracker")}
            if state is not None
            else set()
        )
        if marked.propose not in allowed:
            stage = STAGE_TEXT.get(marked.propose, marked.propose)
            reasons.append(f"переход в «{stage}» трекеру сейчас недоступен")
        elif spec.task.auto_transitions and marked.propose == "validation":
            try:
                repo.apply_transition(
                    conn,
                    task_id,
                    marked.propose,
                    actor="tracker",
                    seq=marked.covered_seq,
                    expected_revision=revision,
                )
            except TaskError as error:
                reasons.append(error.message)
        else:
            proposed = marked.propose
    repo.set_task_covered(conn, task_id, marked.covered_seq)
    _record_tracker_usage(conn, session_id, spec, marked, invariant_hash)
    events: list[dict] = [
        {
            "type": "task",
            "task": _task_payload(conn, session_id),
            "proposed": proposed,
        }
    ]
    if reasons:
        repo.record_task_rejection(
            conn,
            task_id,
            actor="tracker",
            reason="; ".join(reasons),
            seq=marked.covered_seq,
        )
        events.append({"type": "task_rejected", "reasons": reasons})
    return events


def _owns(proposal: dict, owner: dict) -> bool:
    column, value = next(iter(owner.items()))
    return proposal[column] == value


def _items_tokens(items: dict[str, dict]) -> int:
    """A version is measured by what it stores, whoever wrote it."""
    if not items:
        return 0
    return max(1, len(json.dumps(items, ensure_ascii=False)) // 3)


def _record_memory_usage(
    conn: sqlite3.Connection,
    session_id: str,
    spec: AgentSpec,
    extracted: MemoryResult,
    invariant_hash: str | None = None,
) -> None:
    _record_helper_usage(
        conn,
        session_id,
        spec,
        spec.memory.model,
        extracted,
        kind="memory",
        user_profile_hash=extracted.user_profile_hash,
        invariant_hash=invariant_hash,
    )


def _record_tracker_usage(
    conn: sqlite3.Connection,
    session_id: str,
    spec: AgentSpec,
    marked: TrackerResult,
    invariant_hash: str | None = None,
) -> None:
    _record_helper_usage(
        conn,
        session_id,
        spec,
        spec.task.model,
        marked,
        kind="task",
        invariant_hash=invariant_hash,
    )


def _record_invariant_usage(
    conn: sqlite3.Connection,
    session_id: str,
    spec: AgentSpec,
    checked: InvariantChecked,
    invariant_hash: str,
) -> None:
    repo.record_usage(
        conn,
        session_id,
        checked.model,
        Usage(
            tokens_in=int(checked.usage.get("tokens_in", 0)),
            tokens_out=int(checked.usage.get("tokens_out", 0)),
            reasoning=int(checked.usage.get("reasoning", 0)),
            cost=float(checked.usage.get("cost_usd", 0)),
        ),
        checked.latency_ms,
        spec_hash=spec.hash,
        estimate={"input_tokens": checked.input_estimate_tokens},
        context={
            "estimated_tokens": checked.input_estimate_tokens,
            "input_tokens": int(checked.usage.get("tokens_in", 0)),
            "output_tokens": int(checked.usage.get("tokens_out", 0)),
        },
        kind="invariant",
        invariant_hash=invariant_hash,
    )


def _record_helper_usage(
    conn: sqlite3.Connection,
    session_id: str,
    spec: AgentSpec,
    model: str,
    result,
    *,
    kind: str,
    user_profile_hash: str | None = None,
    invariant_hash: str | None = None,
) -> None:
    """Both helper calls are billed the same way; only the `kind` differs."""
    repo.record_usage(
        conn,
        session_id,
        model,
        result.usage,
        result.latency_ms,
        spec_hash=spec.hash,
        estimate={"input_tokens": result.input_estimate_tokens},
        context={
            "estimated_tokens": result.input_estimate_tokens,
            "input_tokens": result.usage.tokens_in,
            "output_tokens": result.usage.tokens_out,
        },
        kind=kind,
        user_profile_hash=user_profile_hash,
        invariant_hash=invariant_hash,
    )


def _sse(event) -> str:
    return _json_sse(dataclasses.asdict(event))


def _json_sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _record_turn_usage(
    conn: sqlite3.Connection,
    session_id: str,
    agent: Agent,
    event: UsageReported,
    message_id: int | None,
    invariant_hash: str | None = None,
) -> None:
    spec = agent.spec
    repo.record_usage(
        conn,
        session_id,
        spec.model.model,
        Usage(
            tokens_in=event.actual["input_tokens"],
            tokens_out=event.actual["output_tokens"],
            reasoning=event.actual["reasoning_tokens"],
            cost=event.actual["cost_usd"],
        ),
        event.actual["latency_ms"],
        spec_hash=spec.hash,
        message_id=message_id,
        estimate=event.estimate,
        context=event.context,
        user_profile_hash=agent.user_profile_hash,
        invariant_hash=invariant_hash,
    )


async def _model_context_window(request: Request, spec: AgentSpec) -> int | None:
    """Resolve the real ceiling without making catalog availability block chat."""
    if spec.context.window_tokens is not None:
        return None
    try:
        return await request.app.state.providers.context_window(
            spec.model.provider, spec.model.model
        )
    except Exception:
        return None


def _store_tool_call(
    conn: sqlite3.Connection,
    session_id: str,
    event: ToolFinished,
    started: dict[str, ToolCall],
) -> None:
    call = started.pop(
        event.call_id, ToolCall(id=event.call_id, tool=event.tool, arguments_json="{}")
    )
    status = "ok" if event.ok else ("rejected" if event.output == REFUSED else "error")
    repo.record_tool_call(
        conn,
        session_id,
        call,
        ToolResult(
            call_id=event.call_id,
            output=event.output,
            ok=event.ok,
            exit_code=event.exit_code,
            duration_ms=event.duration_ms,
        ),
        status,
    )
