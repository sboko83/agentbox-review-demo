"""The tracker: a cheap call that marks what a turn did to the task.

A separate call rather than one more field in the memory router's answer: the
two have different contracts and different trigger conditions — the router
lives on a session without a task, the tracker on a session without memory —
and a failure of one must not cancel the other.

The tracker never gets the last word. It returns a decision; the state machine
accepts or refuses it, so a step outside the plan, a verdict outside the
validation stage and a transition whose guard is unmet are refused in the very
place a human's own move would be refused.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from agentbox.core.context import (
    HELPER_REASONING_HEADROOM,
    _empty_helper_error,
    _line,
)
from agentbox.core.ledger import Ledger
from agentbox.core.invariants import InvariantSet
from agentbox.core.ports import LLMError, LLMPort, LLMRequest, Usage
from agentbox.core.task_machine import (
    STAGE_TEXT,
    TaskState,
    available_transitions,
    render,
)

TRACKER_STAGES = ("execution", "validation")
TRACKER_FORMAT = {"type": "json_object"}

TRACKER_ROLE = (
    "Ты трекер задачи. По состоянию задачи и непрочитанной части диалога отметь,"
    " что уже произошло. Ничего не выполняй, не советуй и не планируй.\n"
    'Верни ровно один JSON-объект: {"step": {"index": число, "status":'
    ' "todo|in_progress|done|blocked", "note": "коротко"} | null, "verdict":'
    ' "pass|fail|null", "propose": "этап|null"}.\n'
    "Шаг помечай done, только если из диалога видно, что он действительно"
    " сделан; blocked — когда работа встала и нужен человек, тогда note"
    " обязателен и называет причину.\n"
    'Если менять нечего, верни {"step": null, "verdict": null, "propose": null}.'
)

# The verdict belongs to the validation stage alone, so the role says so
# instead of letting the machine refuse the same field turn after turn.
STAGE_RULE = {
    "execution": (
        "Сейчас этап выполнения: отмечай только шаг, verdict всегда null."
    ),
    "validation": (
        "Сейчас этап проверки: шаги уже не двигаются, вынеси verdict — pass,"
        " если сошлись все критерии, fail, если хотя бы один нет."
    ),
}

_NULLS = {"", "null", "none", "нет"}


@dataclass(frozen=True, slots=True)
class TrackerStep:
    """A mark on one step; the machine decides whether it lands."""

    index: int
    status: str | None = None
    note: str | None = None


@dataclass(frozen=True, slots=True)
class TrackerResult:
    covered_seq: int
    step: TrackerStep | None = None
    verdict: str | None = None
    propose: str | None = None
    rejected: list[str] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    input_estimate_tokens: int = 0
    latency_ms: int = 0
    error: str | None = None


def tracker_runs(state: TaskState | None) -> bool:
    """The tracker watches a task that is being worked on, and nothing else.

    On `planning` there is nothing to mark yet, on a pause nothing may move,
    and a closed task is history.
    """
    return (
        state is not None and not state.paused and state.stage in TRACKER_STAGES
    )


@dataclass(frozen=True, slots=True)
class TaskTracker:
    """Turns a finished turn into a step mark, a verdict or a proposed stage."""

    model: str
    max_tokens: int

    async def observe(
        self,
        llm: LLMPort,
        state: TaskState,
        history: list[dict],
        ledger: Ledger,
        *,
        covered_seq: int = 0,
        invariants: InvariantSet | None = None,
    ) -> TrackerResult | None:
        """Read the dialogue past `covered_seq` and say what it did to the task."""
        if not tracker_runs(state):
            return None
        pending = [entry for entry in history if int(entry["seq"]) > covered_seq]
        if not pending:
            return None
        allowed = tuple(item.stage for item in available_transitions(state, "tracker"))
        stages = (
            "Доступные для предложения этапы: "
            + ", ".join(f"{stage} ({STAGE_TEXT[stage]})" for stage in allowed)
            if allowed
            else "Предлагать смену этапа сейчас нечего: верни propose = null."
        )
        role = "\n".join([TRACKER_ROLE, STAGE_RULE[state.stage], stages])
        prompt = (
            f"Состояние задачи:\n{render(state)}\n\n"
            "Непрочитанная часть диалога:\n"
            + "\n".join(_line(entry) for entry in pending)
        )
        messages = [{"role": "system", "content": role}]
        if invariants and invariants.rules:
            messages.append({"role": "system", "content": invariants.render()})
        messages.append({"role": "user", "content": prompt})
        request = LLMRequest(
            model=self.model,
            messages=messages,
            max_output_tokens=self.max_tokens + HELPER_REASONING_HEADROOM,
            text_format=TRACKER_FORMAT,
            tools=[],
            reasoning_effort="low",
        )
        ledger.check_budget()
        parts: list[str] = []
        usage = Usage()
        started_at = time.perf_counter()
        async for chunk in llm.stream(request):
            if chunk.text:
                parts.append(chunk.text)
            if chunk.usage is not None:
                usage = chunk.usage
        ledger.record(usage)
        step: TrackerStep | None = None
        verdict: str | None = None
        propose: str | None = None
        rejected: list[str] = []
        error: str | None = None
        try:
            if not "".join(parts).strip():
                raise LLMError(
                    _empty_helper_error("пустой ход трекера", usage, self.max_tokens)
                )
            step, verdict, propose, rejected = _parse("".join(parts))
        except LLMError as parse_error:
            error = str(parse_error)
        return TrackerResult(
            covered_seq=int(history[-1]["seq"]),
            step=step,
            verdict=verdict,
            propose=propose,
            rejected=rejected,
            usage=usage,
            input_estimate_tokens=max(
                1,
                (
                    len(role)
                    + len(prompt)
                    + (len(invariants.render()) if invariants and invariants.rules else 0)
                )
                // 3,
            ),
            latency_ms=round((time.perf_counter() - started_at) * 1000),
            error=error,
        )


def _parse(text: str) -> tuple[TrackerStep | None, str | None, str | None, list[str]]:
    """Read the answer's shape only; the meaning is the machine's to judge."""
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise LLMError(f"модель вернула ход трекера не в JSON: {error.msg}") from error
    if not isinstance(payload, dict):
        raise LLMError("модель вернула ход трекера не как JSON-объект")
    rejected: list[str] = []
    step = _step(payload.get("step"), rejected)
    verdict = _word(payload.get("verdict"), "вердикт", rejected)
    propose = _word(payload.get("propose"), "предложенный этап", rejected)
    return step, verdict, propose, rejected


def _step(raw: object, rejected: list[str]) -> TrackerStep | None:
    if raw is None:
        return None
    if isinstance(raw, str) and raw.strip().lower() in _NULLS:
        return None
    if not isinstance(raw, dict):
        rejected.append("шаг: ответ не описывает шаг объектом")
        return None
    if not raw:
        return None
    index = raw.get("index")
    if isinstance(index, bool) or not isinstance(index, int):
        rejected.append(f"шаг: индекс {index!r} не число")
        return None
    status = _word(raw.get("status"), "статус шага", rejected)
    note = _word(raw.get("note"), "заметка шага", rejected)
    return TrackerStep(index=index, status=status, note=note)


def _word(raw: object, what: str, rejected: list[str]) -> str | None:
    """A missing value and the string «null» mean the same: nothing to say."""
    if raw is None:
        return None
    if not isinstance(raw, str):
        rejected.append(f"{what}: {raw!r} не строка")
        return None
    value = raw.strip()
    return None if value.lower() in _NULLS else value
