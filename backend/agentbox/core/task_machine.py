"""Task state as a finite state machine: stages, guards, expected action.

A pure module: no database, no HTTP, no model call. Routes and the tracker
only ask it what is allowed and what the next state looks like, so an illegal
transition is refused in one place and every pair (stage, transition) is
testable without a running application.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, replace
from typing import Literal, Sequence

Stage = Literal["planning", "execution", "validation", "done", "cancelled"]
StepStatus = Literal["todo", "in_progress", "done", "blocked"]
Verdict = Literal["pass", "fail"]
Actor = Literal["user", "tracker"]

STAGES: tuple[Stage, ...] = ("planning", "execution", "validation", "done", "cancelled")
TERMINAL_STAGES: tuple[Stage, ...] = ("done", "cancelled")
STEP_STATUSES: tuple[StepStatus, ...] = ("todo", "in_progress", "done", "blocked")

MAX_STEPS = 30
MAX_CRITERIA = 30
MAX_TITLE_CHARS = 200
MAX_NOTE_CHARS = 500

STAGE_TEXT = {
    "planning": "планирование",
    "execution": "выполнение",
    "validation": "проверка",
    "done": "завершена",
    "cancelled": "отменена",
}
STEP_STATUS_TEXT = {
    "todo": "ждёт",
    "in_progress": "в работе",
    "done": "готов",
    "blocked": "заблокирован",
}

# The role suffix and the tool list of a stage travel together: the server
# assembles both from the task row, so no text in the UI can widen what the
# agent may do on `planning`, on `validation` or while the task is paused.
PLANNING_ROLE = (
    "Сейчас включён режим планирования. Составляй и уточняй выполнимый план, "
    "но не предлагай и не вызывай инструменты. Ничего не выдавай за уже "
    "выполненное: выполнение начнётся только после явного перехода пользователя."
)
EXECUTION_ROLE = (
    "Задача в работе. Делай текущий шаг и не забегай вперёд: закончив его, "
    "скажи, что сделано, и остановись. Просьба пропустить этап не меняет "
    "состояние: назови невыполненное условие и допустимое следующее действие."
)
VALIDATION_ROLE = (
    "Этап проверки. Сверь результат с критериями и ничего не меняй: запись "
    "в файлы и произвольные команды на этом этапе запрещены. Просьба завершить "
    "задачу без актуального pass не меняет состояние: объясни допустимый путь."
)
PAUSED_ROLE = (
    "Задача на паузе. Отвечай на вопросы и обсуждай, но шаги не выполняй "
    "и инструменты не вызывай, пока человек не продолжит задачу."
)
STAGE_ROLE: dict[Stage, str] = {
    "planning": PLANNING_ROLE,
    "execution": EXECUTION_ROLE,
    "validation": VALIDATION_ROLE,
}

TASK_HEADER = (
    "Состояние задачи — продолжай с текущего шага. Цель, план и сделанное уже"
    " здесь, переспрашивать их не нужно."
)

# Who may drive which transition. A transition that hands out rights — tools on
# `execution`, a closed working layer on `done` — belongs to the human alone;
# the tracker can only propose it.
ALLOWED_ACTORS: dict[tuple[Stage, Stage], tuple[Actor, ...]] = {
    ("planning", "execution"): ("user",),
    ("planning", "planning"): ("user",),
    ("planning", "cancelled"): ("user",),
    ("execution", "validation"): ("user", "tracker"),
    ("execution", "planning"): ("user",),
    ("execution", "cancelled"): ("user",),
    ("validation", "done"): ("user",),
    ("validation", "execution"): ("user", "tracker"),
    ("validation", "planning"): ("user",),
    ("validation", "cancelled"): ("user",),
}


class TaskError(ValueError):
    """A refused transition carries a code: the UI explains, the test asserts."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True, slots=True)
class Step:
    index: int
    title: str
    status: StepStatus = "todo"
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "title": self.title,
            "status": self.status,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "Step":
        return cls(
            index=int(raw["index"]),
            title=str(raw.get("title", "")),
            status=raw.get("status", "todo"),
            note=str(raw.get("note", "") or ""),
        )


@dataclass(frozen=True, slots=True)
class ExpectedAction:
    actor: Literal["user", "agent"]
    text: str

    def to_dict(self) -> dict:
        return {"actor": self.actor, "text": self.text}


@dataclass(frozen=True, slots=True)
class Transition:
    """One button of the task strip: the stage, and what it costs to get there."""

    stage: Stage
    needs_force: bool = False
    reason: str | None = None
    allowed: bool = True
    code: str | None = None

    def to_dict(self) -> dict:
        return {
            "stage": self.stage,
            "needs_force": self.needs_force,
            "reason": self.reason,
            "allowed": self.allowed,
        }


@dataclass(frozen=True, slots=True)
class TaskState:
    stage: Stage = "planning"
    paused: bool = False
    steps: tuple[Step, ...] = ()
    current_step: int | None = None
    criteria: tuple[str, ...] = ()
    verdict: Verdict | None = None
    plan_text: str = ""
    plan_error: str | None = None
    revision: int = 0
    result_revision: int = 0
    approved_plan_hash: str | None = None
    approved_invariant_hash: str | None = None
    verdict_result_revision: int | None = None

    @property
    def is_terminal(self) -> bool:
        return self.stage in TERMINAL_STAGES

    def step(self, index: int) -> Step | None:
        for item in self.steps:
            if item.index == index:
                return item
        return None

    def to_dict(self) -> dict:
        return {
            "stage": self.stage,
            "paused": self.paused,
            "steps": [item.to_dict() for item in self.steps],
            "current_step": self.current_step,
            "criteria": list(self.criteria),
            "verdict": self.verdict,
            "plan_text": self.plan_text,
            "plan_error": self.plan_error,
            "revision": self.revision,
            "result_revision": self.result_revision,
            "approved_plan_hash": self.approved_plan_hash,
            "approved_invariant_hash": self.approved_invariant_hash,
            "verdict_result_revision": self.verdict_result_revision,
            "expected_action": expected_action(self).to_dict(),
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "TaskState":
        return cls(
            stage=raw.get("stage", "planning"),
            paused=bool(raw.get("paused", False)),
            steps=tuple(Step.from_dict(item) for item in raw.get("steps", ())),
            current_step=raw.get("current_step"),
            criteria=tuple(str(item) for item in raw.get("criteria", ())),
            verdict=raw.get("verdict"),
            plan_text=str(raw.get("plan_text", "") or ""),
            plan_error=raw.get("plan_error"),
            revision=int(raw.get("revision", 0)),
            result_revision=int(raw.get("result_revision", 0)),
            approved_plan_hash=raw.get("approved_plan_hash"),
            approved_invariant_hash=raw.get("approved_invariant_hash"),
            verdict_result_revision=raw.get("verdict_result_revision"),
        )


def plan_hash(state: TaskState) -> str:
    """Stable identity of the plan a human approved."""
    payload = {
        "steps": [item.title for item in state.steps],
        "criteria": list(state.criteria),
    }
    body = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def plan_is_approved(state: TaskState) -> bool:
    return bool(state.approved_plan_hash) and state.approved_plan_hash == plan_hash(state)


def _changed(
    state: TaskState,
    *,
    result: bool = False,
    clear_approval: bool = False,
    **changes,
) -> TaskState:
    """Advance optimistic and result versions with every persisted decision."""
    if result:
        changes.setdefault("result_revision", state.result_revision + 1)
        changes.setdefault("verdict", None)
        changes.setdefault("verdict_result_revision", None)
    if clear_approval:
        changes.setdefault("approved_plan_hash", None)
        changes.setdefault("approved_invariant_hash", None)
    changes["revision"] = state.revision + 1
    return replace(state, **changes)


def _renumber(steps: Sequence[Step]) -> tuple[Step, ...]:
    return tuple(replace(item, index=number) for number, item in enumerate(steps, start=1))


def _advance(state: TaskState) -> TaskState:
    """Point at the first unfinished step and mark it as the one in work."""
    target = next(
        (item for item in state.steps if item.status in ("todo", "in_progress")), None
    )
    if target is None:
        return replace(state, current_step=None)
    steps = tuple(
        replace(item, status="in_progress") if item.index == target.index else item
        for item in state.steps
    )
    return replace(state, steps=steps, current_step=target.index)


def _guard(state: TaskState, to_stage: Stage, actor: Actor) -> Transition:
    """Why this transition is refused; strict guards are never forceable."""
    if to_stage == "planning":
        return Transition(to_stage)
    if to_stage == "cancelled":
        return Transition(to_stage)
    if state.stage == "planning" and to_stage == "execution":
        if not state.steps:
            reason = state.plan_error or "План не разобран: нет ни одного шага"
            return Transition(to_stage, reason=reason, allowed=False, code="no_steps")
        return Transition(to_stage)
    if state.stage == "execution" and to_stage == "validation":
        if not plan_is_approved(state):
            return Transition(
                to_stage,
                reason="Текущая версия плана не утверждена: вернитесь в планирование",
                allowed=False,
                code="stale_approval",
            )
        pending = [item for item in state.steps if item.status != "done"]
        if pending:
            return Transition(
                to_stage,
                reason=f"Не выполнены шаги: {', '.join(str(i.index) for i in pending)}",
                allowed=False,
                code="unfinished_steps",
            )
        return Transition(to_stage)
    if state.stage == "validation" and to_stage == "done":
        if not plan_is_approved(state):
            return Transition(
                to_stage,
                reason="Текущая версия плана не утверждена: вернитесь в планирование",
                allowed=False,
                code="stale_approval",
            )
        if state.verdict != "pass":
            return Transition(
                to_stage,
                reason="Вердикт проверки ещё не «pass»",
                allowed=False,
                code="missing_pass",
            )
        if state.verdict_result_revision != state.result_revision:
            return Transition(
                to_stage,
                reason="Результат изменён после проверки: выполните проверку снова",
                allowed=False,
                code="stale_verdict",
            )
        return Transition(to_stage)
    if state.stage == "validation" and to_stage == "execution":
        if state.verdict != "fail":
            return Transition(
                to_stage,
                reason="Проверка не признала результат неудачным",
                allowed=False,
                code="missing_fail",
            )
        return Transition(to_stage)
    return Transition(to_stage)


def available_transitions(state: TaskState, actor: Actor = "user") -> tuple[Transition, ...]:
    """The transitions the strip may draw, including blocked moves and reasons."""
    if state.is_terminal:
        return ()
    if state.paused:
        # Everything but giving up waits for a resume: the stage a task
        # returns to must be the stage it stopped on.
        return (Transition("cancelled"),) if actor == "user" else ()
    result: list[Transition] = []
    for (from_stage, to_stage), actors in ALLOWED_ACTORS.items():
        if from_stage != state.stage or actor not in actors:
            continue
        option = _guard(state, to_stage, actor)
        if not option.allowed and actor != "user":
            continue
        result.append(option)
    return tuple(sorted(result, key=lambda item: STAGES.index(item.stage)))


def transition(
    state: TaskState,
    to_stage: Stage,
    actor: Actor = "user",
    *,
    force: bool = False,
    failed_criteria: Sequence[str] = (),
    invariant_hash: str | None = None,
) -> TaskState:
    """Apply a stage change or raise `TaskError` naming the guard that refused."""
    if to_stage not in STAGES:
        raise TaskError("stage", f"Неизвестный этап: {to_stage}")
    if force:
        raise TaskError("force_removed", "Принудительные переходы отключены")
    if state.is_terminal:
        raise TaskError("terminal", "Задача уже закрыта")
    if state.paused and to_stage != "cancelled":
        raise TaskError("paused", "Задача на паузе: сначала продолжить")
    actors = ALLOWED_ACTORS.get((state.stage, to_stage))
    if actors is None:
        raise TaskError(
            "transition",
            f"Переход «{STAGE_TEXT[state.stage]} → {STAGE_TEXT[to_stage]}» не предусмотрен",
        )
    if actor not in actors:
        raise TaskError("actor", "Этот переход выполняет только человек")
    option = _guard(state, to_stage, actor)
    if not option.allowed:
        raise TaskError(option.code or "guard", option.reason or "Условие перехода не выполнено")

    if to_stage == "planning":
        # Replanning keeps nothing but the journal: finished steps stay there.
        return TaskState(stage="planning", paused=False, revision=state.revision + 1)
    if to_stage == "execution":
        if not invariant_hash:
            raise TaskError("invariant_version", "Не указана утверждаемая версия инвариантов")
        steps = state.steps
        if state.stage == "validation":
            if state.criteria and not failed_criteria:
                raise TaskError(
                    "failed_criteria", "Укажите критерии, которые не прошли проверку"
                )
            extra = [
                Step(index=0, title=_clean(item)[:MAX_TITLE_CHARS])
                for item in failed_criteria
                if _clean(item)
            ]
            steps = _renumber([*steps, *extra])
        moved = _changed(
            state,
            stage="execution",
            steps=steps,
            verdict=None,
            verdict_result_revision=None,
            approved_invariant_hash=invariant_hash,
        )
        moved = replace(moved, approved_plan_hash=plan_hash(moved))
        return _advance(moved)
    if to_stage == "validation":
        return _changed(
            state,
            stage="validation",
            verdict=None,
            verdict_result_revision=None,
            current_step=None,
        )
    return _changed(state, stage=to_stage, paused=False, current_step=None)


def set_paused(state: TaskState, paused: bool, actor: Actor = "user") -> TaskState:
    """Pause is a flag, not a stage: resuming returns the very stage it stopped on."""
    if actor != "user":
        raise TaskError("actor", "Паузу ставит и снимает человек")
    if state.is_terminal:
        raise TaskError("terminal", "Задача уже закрыта")
    if state.paused == paused:
        raise TaskError(
            "paused" if paused else "running",
            "Задача уже на паузе" if paused else "Задача не на паузе",
        )
    return _changed(state, paused=paused)


def set_step(
    state: TaskState,
    index: int,
    actor: Actor = "user",
    *,
    status: StepStatus | None = None,
    title: str | None = None,
    note: str | None = None,
) -> TaskState:
    if state.is_terminal:
        raise TaskError("terminal", "Задача уже закрыта")
    if actor == "tracker" and (state.paused or state.stage not in ("execution", "validation")):
        raise TaskError("tracker_idle", "Трекер не ведёт шаги на этом этапе")
    target = state.step(index)
    if target is None:
        raise TaskError("step_index", f"Шага {index} в задаче нет")
    if status is not None and status not in STEP_STATUSES:
        raise TaskError("step_status", f"Неизвестный статус шага: {status}")
    updated = replace(
        target,
        status=status or target.status,
        title=_clean(title)[:MAX_TITLE_CHARS] if title is not None else target.title,
        note=_clean(note)[:MAX_NOTE_CHARS] if note is not None else target.note,
    )
    if not updated.title:
        raise TaskError("step_title", "Шаг не может остаться без названия")
    steps = tuple(updated if item.index == index else item for item in state.steps)
    moved = _changed(
        state,
        steps=steps,
        result=True,
        clear_approval=title is not None and title != target.title,
    )
    if status == "in_progress":
        return replace(moved, current_step=index)
    if status in ("done", "blocked") and state.current_step == index:
        return _advance(moved) if status == "done" else moved
    return moved


def set_verdict(state: TaskState, verdict: Verdict | None, actor: Actor = "tracker") -> TaskState:
    if state.stage != "validation":
        raise TaskError("verdict_stage", "Вердикт выносится только на этапе проверки")
    if state.paused:
        raise TaskError("paused", "Задача на паузе")
    if verdict not in ("pass", "fail", None):
        raise TaskError("verdict", f"Неизвестный вердикт: {verdict}")
    return _changed(
        state,
        verdict=verdict,
        verdict_result_revision=state.result_revision if verdict is not None else None,
    )


def expected_action(state: TaskState) -> ExpectedAction:
    """A function of the state, never a sentence from the model.

    Generated text would drift with the wording and could not be asserted.
    """
    if state.stage == "done":
        return ExpectedAction("user", "задача завершена")
    if state.stage == "cancelled":
        return ExpectedAction("user", "задача отменена")
    if state.paused:
        return ExpectedAction("user", "продолжить задачу")
    if state.stage == "planning":
        if not state.steps:
            return ExpectedAction("agent", "составить план")
        return ExpectedAction("user", "утвердить или уточнить план")
    if state.stage == "execution":
        if not plan_is_approved(state):
            return ExpectedAction("user", "вернуться в планирование и утвердить изменённый план")
        current = state.step(state.current_step) if state.current_step else None
        if current is not None and current.status == "blocked":
            return ExpectedAction(
                "user", f"снять блокер шага {current.index}: «{current.note or current.title}»"
            )
        if current is not None and current.status in ("in_progress", "todo"):
            return ExpectedAction("agent", f"выполнить шаг {current.index} «{current.title}»")
        blocked = next((item for item in state.steps if item.status == "blocked"), None)
        if blocked is not None:
            return ExpectedAction(
                "user", f"снять блокер шага {blocked.index}: «{blocked.note or blocked.title}»"
            )
        pending = next((item for item in state.steps if item.status != "done"), None)
        if pending is not None:
            return ExpectedAction("agent", f"выполнить шаг {pending.index} «{pending.title}»")
        return ExpectedAction("user", "перевести задачу в проверку")
    if not plan_is_approved(state):
        return ExpectedAction("user", "вернуться в планирование и утвердить изменённый план")
    if state.verdict is None:
        return ExpectedAction("agent", "проверить результат по критериям")
    if state.verdict == "pass":
        return ExpectedAction("user", "подтвердить завершение")
    return ExpectedAction("user", "вернуть задачу в выполнение")


# --- plan parsing -----------------------------------------------------------

PLAN_FORMAT = (
    "Оформи план так: нумерованный список шагов, по одному шагу на строку,"
    " затем строка «Проверка:» и список критериев, по одному на строку."
    " Без вложенных списков и без пояснений между шагами."
)

_CRITERIA_HEADER = re.compile(
    r"^[\s*_#>]*(проверка|критерии(\s+проверки)?|checks?|validation)\s*[:：]?[\s*_]*$",
    re.IGNORECASE,
)
_NUMBERED = re.compile(r"^[\s>]*(?:[-*•]\s*)?(\d{1,2})\s*[.)\]:-]\s+(.+)$")
_BULLET = re.compile(r"^[\s>]*[-*•]\s+(.+)$")
_HEADING = re.compile(r"^[\s*_#>]*(план|шаги|plan|steps)\s*[:：]?[\s*_]*$", re.IGNORECASE)


def _clean(value: str | None) -> str:
    if not value:
        return ""
    value = value.replace("**", "").replace("__", "").strip()
    return re.sub(r"\s+", " ", value).strip(" .;")


@dataclass(frozen=True, slots=True)
class ParsedPlan:
    steps: tuple[str, ...]
    criteria: tuple[str, ...]
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def parse_plan(text: str) -> ParsedPlan:
    """Read the planner answer as steps and criteria; tolerate the numbering.

    An unparsed plan stays a draft with a reason instead of failing: the model
    writes it, and the person must be told what to fix.
    """
    steps: list[str] = []
    criteria: list[str] = []
    in_criteria = False
    for raw_line in (text or "").splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            continue
        if _CRITERIA_HEADER.match(line):
            in_criteria = True
            continue
        if _HEADING.match(line):
            continue
        numbered = _NUMBERED.match(line)
        bullet = _BULLET.match(line)
        if numbered:
            item = _clean(numbered.group(2))
        elif bullet:
            item = _clean(bullet.group(1))
        elif in_criteria:
            item = _clean(line)
        else:
            continue
        if not item:
            continue
        target = criteria if in_criteria else steps
        target.append(item[:MAX_TITLE_CHARS])
    if not steps:
        return ParsedPlan((), (), "План не разобран: не найден нумерованный список шагов")
    if len(steps) > MAX_STEPS:
        return ParsedPlan((), (), f"В плане больше {MAX_STEPS} шагов — разбей задачу")
    return ParsedPlan(tuple(steps), tuple(criteria[:MAX_CRITERIA]))


def plan_state(state: TaskState, text: str) -> TaskState:
    """Store the planner answer; steps appear only when the plan parsed."""
    if state.stage != "planning":
        raise TaskError("plan_stage", "План принимается только на этапе планирования")
    parsed = parse_plan(text)
    return _changed(
        state,
        result=True,
        clear_approval=True,
        plan_text=text,
        plan_error=parsed.error,
        steps=_renumber([Step(index=0, title=title) for title in parsed.steps]),
        criteria=parsed.criteria,
        current_step=None,
    )


def render(state: TaskState) -> str:
    """Compact deterministic block: one line per step, stage first."""
    action = expected_action(state)
    lines = [
        f"Этап: {STAGE_TEXT[state.stage]}" + (" (на паузе)" if state.paused else ""),
    ]
    if state.steps:
        lines.append("Шаги:")
        for item in state.steps:
            marker = "→ " if item.index == state.current_step else "  "
            note = f" — {item.note}" if item.note else ""
            lines.append(
                f"{marker}{item.index}. {item.title} [{STEP_STATUS_TEXT[item.status]}]{note}"
            )
    elif state.plan_text:
        lines.append("План ещё не разобран на шаги.")
    if state.criteria:
        lines.append("Критерии проверки:")
        lines.extend(f"- {item}" for item in state.criteria)
    if state.verdict:
        lines.append(f"Вердикт проверки: {state.verdict}")
    lines.append(f"Ожидаемое действие ({'агент' if action.actor == 'agent' else 'человек'}): {action.text}")
    return "\n".join(lines)


def stage_role(state: "TaskState | None") -> str:
    """The role suffix this stage adds, empty for a session without a task."""
    if state is None or state.is_terminal:
        return ""
    if state.paused:
        return PAUSED_ROLE
    return STAGE_ROLE.get(state.stage, "")


def permits_tool(state: "TaskState | None", name: str, mutating: bool) -> bool:
    """Whether this stage lets the agent call that tool at all."""
    if state is None or state.is_terminal:
        return True
    if state.paused or state.stage == "planning":
        return False
    if state.stage == "validation":
        # `run_command` is classified as mutating and no restricted validation
        # executor exists yet, so exposing it would allow shell writes.
        return not mutating
    return plan_is_approved(state)


def block_text(state: "TaskState | None") -> str:
    """The payload block: header plus the rendered state, or nothing."""
    if state is None or state.is_terminal:
        return ""
    return f"{TASK_HEADER}\n{render(state)}"
