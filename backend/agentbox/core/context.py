"""Context strategy port, the built-in strategy plugins and the memory layers.

A strategy owns the short-term layer alone: it selects a view over the
immutable session transcript and never deletes stored history.  The working
and long-term layers are orthogonal to that choice — they arrive through
`ContextState` and compose into the same view, so switching a strategy does
not switch memory on or off.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import time
from typing import Literal, Protocol, get_args, get_origin

from pydantic import ValidationError

from agentbox.core.invariants import InvariantSet
from agentbox.core.ledger import Ledger
from agentbox.core.memory import Conversation, Entry, Message, ToolExchange
from agentbox.core.ports import LLMError, LLMPort, LLMRequest, Usage
from agentbox.core.spec import AgentSpec, ContextSection
from agentbox.core.task_machine import TaskState, block_text as task_block
from agentbox.core.user_profile import UserProfile

SUMMARY_ROLE = (
    "Сжимай историю диалога без домыслов. Сохраняй факты пользователя, "
    "решения, ограничения, результаты инструментов и незавершённые действия. "
    "Убирай повторы и формулируй кратко. Верни только новую сводку."
)
MEMORY_ROLE = (
    "Раскладывай диалог по слоям памяти. Верни один JSON-объект без markdown: "
    "ключ верхнего уровня — имя слоя, значение — объект «ключ памяти → "
    '{"value": строка, "kind": вид}». Ключ памяти — короткое устойчивое имя, '
    "значение — дословная формулировка из диалога, без перевода и "
    "нормализации. Слой называй явно, вид бери из списка своего слоя. "
    "Заполняй каждый слой, для которого в диалоге есть материал, и возвращай "
    "только затронутые ключи: неизменившееся повторять не нужно, опровергнутое "
    "верни со значением null. Ключ называй по сути факта, а не по его виду, "
    "один факт клади в один слой, не заводи второй ключ для того же факта и не "
    "додумывай. Пример ответа: "
    '{"working": {"deadline": {"value": "макет к среде", "kind": "constraint"}}, '
    '"longterm": {"editor": {"value": "работает в Xcode", "kind": "knowledge"}}}'
)
LAYER_ROLE = {
    "working": (
        "Слой working — текущая задача: цель, ограничения, шаги, артефакты, "
        "блокеры. Виды: goal, constraint, step, artifact, blocker."
    ),
    "longterm": (
        "Слой longterm — то, что переживает задачу: кто этот человек (род "
        "занятий, опыт, окружение), принятые решения, знания. Виды: profile, "
        "decision, knowledge."
    ),
}
# The router may change how to answer, never who the person is: the name
# belongs to the list of profiles, `about` is a long-term fact.
ROUTER_PROFILE_FIELDS = (
    "language",
    "tone",
    "verbosity",
    "structure",
    "max_words",
    "code_style",
    "constraints",
)
PROFILE_FIELD_HINT = {
    "max_words": "целое от 10 или null",
    "constraints": "полный новый список строк-правил, включая прежние",
}
PROFILE_PROPOSAL_KIND = "setting"

LAYER_KINDS = {
    "working": ("goal", "constraint", "step", "artifact", "blocker"),
    "longterm": ("profile", "decision", "knowledge"),
}
MEMORY_FORMAT = {
    "type": "json_object",
}
PROFILE_CONTEXT = (
    "Профиль пользователя — как ему отвечать. Он уже учтён в ответах, не "
    "повторяй его значения в слоях памяти:"
)
LONGTERM_HEADER = "Долговременная память — о человеке, решения, знания:"
WORKING_HEADER = "Рабочая память текущей задачи:"
SUMMARY_HEADER = "Сводка предыдущего диалога:"
# Reasoning models count their thinking against `max_output_tokens`; at low
# effort gpt-5-nano can spend more than a thousand tokens before the first
# output token, so the configured helper budget bounds the visible result,
# not the whole call.
HELPER_REASONING_HEADROOM = 2048


@dataclass(frozen=True, slots=True)
class ContextState:
    """Session state shared by context strategy plugins."""

    role: str
    chars_per_token: float
    summary: dict | None = None
    working: dict | None = None
    longterm: dict | None = None
    working_max_tokens: int = 0
    longterm_max_tokens: int = 0
    invariants: InvariantSet | None = None
    user_profile: UserProfile | None = None
    task: TaskState | None = None


@dataclass(slots=True)
class ContextView:
    """Observable provider context produced from a complete transcript."""

    mode: str
    role: str
    tail: list[Entry]
    chars_per_token: float
    summary: dict | None = None
    working: dict | None = None
    longterm: dict | None = None
    working_max_tokens: int = 0
    longterm_max_tokens: int = 0
    invariants: InvariantSet | None = None
    user_profile: UserProfile | None = None
    task: TaskState | None = None
    dropped_seqs: list[int] = field(default_factory=list)
    estimated_tokens: int = 0
    role_tokens: int = 0
    invariant_tokens: int = 0
    profile_tokens: int = 0
    longterm_tokens: int = 0
    working_tokens: int = 0
    task_tokens: int = 0
    summary_tokens: int = 0
    tail_tokens: int = 0
    # Keys the budget left out of this call; the stored layer keeps them.
    longterm_dropped: list[str] = field(default_factory=list)
    working_dropped: list[str] = field(default_factory=list)
    longterm_body: str = ""
    working_body: str = ""
    task_body: str = ""

    def __post_init__(self) -> None:
        self._recalculate()

    @property
    def first_seq(self) -> int | None:
        return self.tail[0].seq if self.tail else None

    def render(self, transport: str) -> list[dict]:
        """Render the selected view in the provider transport's spelling."""
        method = "responses_items" if transport == "responses" else "chat_items"
        # Stable first, volatile last: the profile and the long-term block
        # change least often, so they stay a cacheable prefix across turns.
        items: list[dict] = [{"role": "system", "content": self.role}]
        for block in self._blocks():
            items.append({"role": "system", "content": block})
        for entry in self.tail:
            items.extend(getattr(entry, method)())
        return items

    def drop_oldest(self) -> bool:
        """Drop one complete old user turn, preserving the current turn."""
        next_turn = next(
            (
                index
                for index, entry in enumerate(self.tail[1:], start=1)
                if isinstance(entry, Message) and entry.role == "user"
            ),
            None,
        )
        if next_turn is None:
            return False
        removed = self.tail[:next_turn]
        self.tail = self.tail[next_turn:]
        self.dropped_seqs.extend(entry.seq for entry in removed)
        self.dropped_seqs.sort()
        self._recalculate()
        return True

    def as_dict(
        self,
        *,
        limit_tokens: int | None = None,
        limit_source: str | None = None,
        overflow: str | None = None,
    ) -> dict:
        """JSON-ready view shared by SSE and session detail responses.

        Every layer carries its own `payload_tokens` — what the block costs in
        the next call, not what it cost to produce.  `overhead_tokens` closes
        the gap to `estimated_tokens` when the caller replaced the estimate
        with the measured request: transport envelope and tool schemas belong
        to no layer.
        """
        return {
            "mode": self.mode,
            "role": {"content": self.role, "tokens": self.role_tokens},
            "invariants": (
                {
                    "hash": self.invariants.hash,
                    "rules": [rule.model_dump() for rule in self.invariants.rules],
                    "content": self.invariants.render(),
                    "payload_tokens": self.invariant_tokens,
                }
                if self.invariants and self.invariants.rules
                else None
            ),
            "user_profile": (
                {
                    "name": self.user_profile.name,
                    "hash": self.user_profile.hash,
                    "content": self.user_profile.render(),
                    "payload_tokens": self.profile_tokens,
                }
                if self.user_profile
                else None
            ),
            "summary": _with_payload(self.summary, self.summary_tokens),
            "working": _with_payload(
                self.working, self.working_tokens, self.working_dropped
            ),
            "longterm": _with_payload(
                self.longterm, self.longterm_tokens, self.longterm_dropped
            ),
            "task": (
                {
                    "stage": self.task.stage,
                    "paused": self.task.paused,
                    "content": task_block(self.task),
                    "payload_tokens": self.task_tokens,
                }
                if self.task and not self.task.is_terminal
                else None
            ),
            "tail": {
                "first_seq": self.first_seq,
                "messages": [_entry_dict(entry) for entry in self.tail],
                "tokens": self.tail_tokens,
            },
            "dropped_seqs": self.dropped_seqs,
            "estimated_tokens": self.estimated_tokens,
            "overhead_tokens": max(0, self.estimated_tokens - self._blocks_tokens()),
            "limit_tokens": limit_tokens,
            "limit_source": limit_source if limit_tokens is not None else None,
            "overflow": overflow,
        }

    def _blocks(self) -> list[str]:
        """The system blocks between role and tail, stable layer first.

        Rendering and accounting read the same list, so a block can never be
        sent without being counted or counted without being sent.  The profile
        renders its own header, the memory layers get theirs here.
        """
        blocks: list[str] = []
        if self.invariants and self.invariants.rules:
            blocks.append(self.invariants.render())
        if self.user_profile:
            blocks.append(self.user_profile.render())
        # An emptied layer is a version with no items: it still exists, but
        # sending its header over an empty object would bill a block that says
        # nothing.  A layer squeezed empty by its budget is the same case.
        if self.longterm_body:
            blocks.append(f"{LONGTERM_HEADER}\n{self.longterm_body}")
        if self.working_body:
            blocks.append(f"{WORKING_HEADER}\n{self.working_body}")
        # After the working layer: the task moves more often than its goal.
        if self.task_body:
            blocks.append(self.task_body)
        if self.summary:
            blocks.append(f"{SUMMARY_HEADER}\n{_summary_text(self.summary)}")
        return blocks

    def _blocks_tokens(self) -> int:
        return (
            self.role_tokens
            + self.invariant_tokens
            + self.profile_tokens
            + self.longterm_tokens
            + self.working_tokens
            + self.task_tokens
            + self.summary_tokens
            + self.tail_tokens
        )

    def _recalculate(self) -> None:
        ratio = self.chars_per_token if self.chars_per_token > 0 else 3.0
        self.role_tokens = _tokens(len(self.role), ratio)
        self.invariant_tokens = (
            _tokens(len(self.invariants.render()), ratio)
            if self.invariants and self.invariants.rules
            else 0
        )
        self.profile_tokens = (
            _tokens(len(self.user_profile.render()), ratio) if self.user_profile else 0
        )
        self.longterm_body, self.longterm_dropped, self.longterm_tokens = _fit_layer(
            LONGTERM_HEADER, self.longterm, self.longterm_max_tokens, ratio
        )
        self.working_body, self.working_dropped, self.working_tokens = _fit_layer(
            WORKING_HEADER, self.working, self.working_max_tokens, ratio
        )
        self.task_body = task_block(self.task)
        self.task_tokens = _tokens(len(self.task_body), ratio) if self.task_body else 0
        self.summary_tokens = (
            _tokens(len(f"{SUMMARY_HEADER}\n{_summary_text(self.summary)}"), ratio)
            if self.summary
            else 0
        )
        self.tail_tokens = _tokens(sum(entry.chars() for entry in self.tail), ratio)
        self.estimated_tokens = self._blocks_tokens()


class ContextStrategy(Protocol):
    """Select the provider-visible context from complete session history."""

    def build(self, history: list[Entry], state: ContextState) -> ContextView:
        ...


@dataclass(frozen=True, slots=True)
class NoneContextStrategy:
    mode: str = "none"

    def build(self, history: list[Entry], state: ContextState) -> ContextView:
        return _view(self.mode, history, state)


@dataclass(frozen=True, slots=True)
class WindowContextStrategy:
    keep_last_messages: int

    def build(self, history: list[Entry], state: ContextState) -> ContextView:
        start = max(0, len(history) - self.keep_last_messages)
        start = _turn_boundary(history, start)
        return _view(
            "window",
            history[start:],
            state,
            dropped=[entry.seq for entry in history[:start]],
        )


@dataclass(frozen=True, slots=True)
class TokensContextStrategy:
    memory_limit_tokens: int

    def build(self, history: list[Entry], state: ContextState) -> ContextView:
        view = _view("tokens", history, state)
        while view.estimated_tokens > self.memory_limit_tokens and view.drop_oldest():
            pass
        return view


@dataclass(frozen=True, slots=True)
class SummaryResult:
    covered_seq: int
    message_count: int
    text: str
    tokens: int
    usage: Usage
    input_estimate_tokens: int
    latency_ms: int


@dataclass(frozen=True, slots=True)
class SummaryContextStrategy:
    keep_last_messages: int
    summary_every: int
    summary_model: str
    summary_max_tokens: int

    def build(self, history: list[Entry], state: ContextState) -> ContextView:
        covered_seq = int(state.summary.get("covered_seq", 0)) if state.summary else 0
        tail = [entry for entry in history if entry.seq > covered_seq]
        return _view("summary", tail, state, summary=state.summary)

    async def summarize_if_due(
        self,
        llm: LLMPort,
        history: list[dict],
        previous: dict | None,
        ledger: Ledger,
    ) -> SummaryResult | None:
        """Compress a complete old turn without changing stored transcript rows."""
        covered_seq = int(previous["covered_seq"]) if previous else 0
        pending = [entry for entry in history if int(entry["seq"]) > covered_seq]
        keep_from_history = max(0, self.keep_last_messages - 1)
        boundary = max(0, len(pending) - keep_from_history)
        boundary = _dict_turn_boundary(pending, boundary)
        eligible = pending[:boundary]
        if len(eligible) < self.summary_every:
            return None

        transcript = "\n".join(_line(entry) for entry in eligible)
        prior = previous["text"] if previous else "(сводки ещё нет)"
        prompt = f"Предыдущая сводка:\n{prior}\n\nНовые сообщения:\n{transcript}"
        request = LLMRequest(
            model=self.summary_model,
            messages=[
                {"role": "system", "content": SUMMARY_ROLE},
                {"role": "user", "content": prompt},
            ],
            max_output_tokens=self.summary_max_tokens + HELPER_REASONING_HEADROOM,
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
        text = "".join(parts).strip()
        if not text:
            raise LLMError(_empty_helper_error("пустую сводку", usage, self.summary_max_tokens))
        ledger.record(usage)
        return SummaryResult(
            covered_seq=int(eligible[-1]["seq"]),
            message_count=(int(previous["message_count"]) if previous else 0)
            + len(eligible),
            text=text,
            tokens=max(0, usage.tokens_out - usage.reasoning),
            usage=usage,
            input_estimate_tokens=max(1, (len(SUMMARY_ROLE) + len(prompt)) // 3),
            latency_ms=round((time.perf_counter() - started_at) * 1000),
        )


@dataclass(frozen=True, slots=True)
class MemoryUpdate:
    """One layer's share of a router answer.

    `items` is the whole layer after the merge, ready to become a version;
    `changes` is only what this turn touched, which is what a candidate list
    shows a human deciding whether the long-term layer should keep it.
    """

    items: dict[str, dict]
    changes: dict[str, dict]


@dataclass(frozen=True, slots=True)
class MemoryResult:
    covered_seq: int
    updates: dict[str, MemoryUpdate] | None
    rejected: list[str]
    usage: Usage
    input_estimate_tokens: int
    latency_ms: int
    error: str | None = None
    user_profile_hash: str | None = None


@dataclass(frozen=True, slots=True)
class MemoryRouter:
    """Turns a finished turn into memory items, saying which layer they join.

    It is a separate cheap call rather than a tool the answering model may
    invoke: memory must be filled whether or not tools are enabled, and the
    routing decision has to be auditable outside the model's own answer.  The
    layer is named by the router, never guessed here — a model that answers
    without naming one has not routed anything.
    """

    model: str
    max_tokens: int
    layers: tuple[str, ...] = ("working",)

    async def extract(
        self,
        llm: LLMPort,
        history: list[dict],
        previous: dict[str, dict | None],
        ledger: Ledger,
        *,
        covered_seq: int = 0,
        user_profile: UserProfile | None = None,
    ) -> MemoryResult | None:
        """Merge the part of the dialogue past `covered_seq` into every layer.

        The user profile is shown read-only, so a declared preference is not
        stored a second time as a long-term fact.
        """
        if not history or not self.layers:
            return None
        pending = [entry for entry in history if int(entry["seq"]) > covered_seq]
        if not pending:
            return None
        transcript = "\n".join(_line(entry) for entry in pending)
        sections = [
            _profile_role() if layer == "profile" else LAYER_ROLE[layer]
            for layer in self.layers
        ]
        current = "\n\n".join(
            f"Слой profile, текущие значения:\n{_profile_json(user_profile)}"
            if layer == "profile" and user_profile is not None
            else f"Слой {layer}, текущая память:\n"
            + json.dumps(
                _layer_values(previous.get(layer)),
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
            for layer in self.layers
        )
        role = "\n".join([MEMORY_ROLE, *sections])
        empty = json.dumps({layer: {} for layer in self.layers}, ensure_ascii=False)
        role += (
            "\nЕсли в диалоге нет изменений памяти, всё равно назови каждый "
            f"доступный слой и верни ровно {empty}; не возвращай пустой корневой объект."
        )
        if len(self.layers) == 1:
            layer = self.layers[0]
            role += (
                f"\nВ этом вызове доступен один слой: обязательный ключ верхнего "
                f"уровня — {layer}. Не возвращай элементы памяти прямо в корне. "
                f"Если изменений нет, верни ровно {empty}."
            )
        if user_profile is not None and "profile" not in self.layers:
            current = (
                f"{PROFILE_CONTEXT}\n{_profile_json(user_profile)}\n\n{current}"
            )
        prompt = f"{current}\n\nДиалог для обновления:\n{transcript}"
        request = LLMRequest(
            model=self.model,
            messages=[
                {"role": "system", "content": role},
                {"role": "user", "content": prompt},
            ],
            max_output_tokens=self.max_tokens + HELPER_REASONING_HEADROOM,
            text_format=MEMORY_FORMAT,
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
        last_seq = int(history[-1]["seq"])
        updates: dict[str, MemoryUpdate] | None = None
        rejected: list[str] = []
        error = None
        try:
            if not "".join(parts).strip():
                raise LLMError(
                    _empty_helper_error("пустую память", usage, self.max_tokens)
                )
            routed, rejected = _parse_layers("".join(parts), self.layers)
            updates = {}
            for layer, changes in routed.items():
                if layer == "profile":
                    accepted, refused = _profile_changes(user_profile, changes)
                    rejected.extend(refused)
                    updates[layer] = MemoryUpdate(items={}, changes=accepted)
                    continue
                items, accepted, refused = _merge_layer(
                    _layer_items(previous.get(layer)), changes, last_seq, layer
                )
                rejected.extend(refused)
                updates[layer] = MemoryUpdate(items=items, changes=accepted)
        except LLMError as parse_error:
            updates = None
            error = str(parse_error)
        return MemoryResult(
            covered_seq=last_seq,
            updates=updates,
            rejected=rejected,
            usage=usage,
            input_estimate_tokens=max(1, (len(role) + len(prompt)) // 3),
            latency_ms=round((time.perf_counter() - started_at) * 1000),
            error=error,
            user_profile_hash=user_profile.hash if user_profile else None,
        )


def strategy_for(context: ContextSection) -> ContextStrategy:
    """Build the configured context plugin."""
    if context.mode == "window":
        return WindowContextStrategy(context.keep_last_messages)
    if context.mode == "tokens":
        return TokensContextStrategy(context.memory_limit_tokens)
    if context.mode == "summary":
        return SummaryContextStrategy(
            context.keep_last_messages,
            context.summary_every,
            context.summary_model,
            context.summary_max_tokens,
        )
    return NoneContextStrategy(mode=context.mode)


def build_context_view(
    spec: AgentSpec,
    history: list[dict],
    *,
    role: str,
    chars_per_token: float,
    summary: dict | None = None,
    working: dict | None = None,
    longterm: dict | None = None,
    invariants: InvariantSet | None = None,
    user_profile: UserProfile | None = None,
    task: TaskState | None = None,
    limit_tokens: int | None = None,
) -> ContextView:
    """Build the next-call view for read APIs without a provider-backed Agent."""
    conversation = Conversation(role=role, limit_tokens=2_000_000)
    conversation.chars_per_token = chars_per_token
    for entry in history:
        if entry["role"] == "tool":
            conversation.add_tool(
                call_id=entry["call_id"],
                tool=entry["tool"],
                arguments_json=entry["arguments_json"],
                output=entry["content"],
                seq=entry["seq"],
            )
        else:
            conversation.add(entry["role"], entry["content"], seq=entry["seq"])
    view = strategy_for(spec.context).build(
        conversation.messages,
        ContextState(
            role=role,
            chars_per_token=chars_per_token,
            summary=summary,
            working=working,
            longterm=longterm,
            working_max_tokens=spec.memory.working_max_tokens,
            longterm_max_tokens=spec.memory.longterm_max_tokens,
            invariants=invariants,
            user_profile=user_profile,
            task=task,
        ),
    )
    if limit_tokens is not None and spec.context.overflow == "drop":
        while view.estimated_tokens > limit_tokens and view.drop_oldest():
            pass
    return view


def _view(
    mode: str,
    history: list[Entry],
    state: ContextState,
    *,
    summary: dict | None = None,
    dropped: list[int] | None = None,
) -> ContextView:
    """Every strategy composes the same layers; only the tail is its own."""
    return ContextView(
        mode=mode,
        role=state.role,
        tail=list(history),
        chars_per_token=state.chars_per_token,
        summary=summary,
        working=state.working,
        longterm=state.longterm,
        working_max_tokens=state.working_max_tokens,
        longterm_max_tokens=state.longterm_max_tokens,
        invariants=state.invariants,
        user_profile=state.user_profile,
        task=state.task,
        dropped_seqs=list(dropped or []),
    )


def _turn_boundary(history: list[Entry], start: int) -> int:
    """Move a slice boundary back to the user that started that turn."""
    while start > 0:
        entry = history[start]
        if isinstance(entry, Message) and entry.role == "user":
            break
        start -= 1
    return start


def _dict_turn_boundary(history: list[dict], start: int) -> int:
    while start > 0 and start < len(history) and history[start]["role"] != "user":
        start -= 1
    return start


def _entry_dict(entry: Entry) -> dict:
    if isinstance(entry, Message):
        return {"seq": entry.seq, "role": entry.role, "content": entry.content}
    return {
        "seq": entry.seq,
        "role": "tool",
        "call_id": entry.call_id,
        "tool": entry.tool,
        "arguments_json": entry.arguments_json,
        "content": entry.output,
    }


def _tokens(chars: int, chars_per_token: float) -> int:
    return math.ceil(chars / chars_per_token) if chars else 0


def _line(entry: dict) -> str:
    if entry["role"] == "tool":
        return (
            f"[tool {entry['tool']}] args={entry['arguments_json']} "
            f"result={entry['content']}"
        )
    label = "Пользователь" if entry["role"] == "user" else "Ассистент"
    return f"{label}: {entry['content']}"


def _layer_items(layer: dict | None) -> dict[str, dict]:
    """Unwrap a stored memory version into its `key -> item` mapping."""
    if not layer:
        return {}
    items = layer.get("items", layer)
    return items if isinstance(items, dict) else {}


def _layer_values(layer: dict | None) -> dict[str, str]:
    return {key: str(item.get("value", "")) for key, item in _layer_items(layer).items()}


def _values_body(items: dict[str, dict]) -> str:
    """The items exactly as the payload spells them."""
    return json.dumps(
        {key: str(item.get("value", "")) for key, item in items.items()},
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    )


def _fit_layer(
    header: str, layer: dict | None, budget: int, chars_per_token: float
) -> tuple[str, list[str], int]:
    """The layer as this call carries it, cut down to its token budget.

    A layer grows with every routed turn while its block is paid for on every
    call, so the budget belongs here and not to the router's good manners.
    What leaves first is the oldest item: the keys the current task just
    routed are the ones the answer is about. Nothing is deleted — the stored
    version keeps every key, only this call's copy is shorter.
    """
    kept = dict(_layer_items(layer))
    dropped: list[str] = []
    while kept:
        body = _values_body(kept)
        tokens = _tokens(len(f"{header}\n{body}"), chars_per_token)
        if budget <= 0 or tokens <= budget:
            return body, dropped, tokens
        oldest = min(kept, key=lambda key: (int(kept[key].get("seq") or 0), key))
        dropped.append(oldest)
        kept.pop(oldest)
    return "", dropped, 0


def _profile_json(profile: UserProfile) -> str:
    return json.dumps(
        profile.model_dump(mode="json", exclude={"name"}),
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    )


def _profile_role() -> str:
    fields = []
    for name in ROUTER_PROFILE_FIELDS:
        annotation = UserProfile.model_fields[name].annotation
        allowed = (
            "|".join(get_args(annotation))
            if get_origin(annotation) is Literal
            else PROFILE_FIELD_HINT[name]
        )
        fields.append(f"{name}: {allowed}")
    return (
        "Слой profile — как отвечать этому человеку: язык, тон, объём, форма, "
        "код, ограничения. Меняй его только по явной просьбе пользователя о "
        "будущих ответах («впредь отвечай по-английски», «пиши короче»); "
        "разовая просьба к одному ответу и сведения о самом человеке сюда не "
        "идут. Правило для всех будущих ответов («впредь», «всегда», «никогда») "
        "— это profile, а не constraint слоя working: working хранит только "
        "ограничения текущей задачи. Кто человек, чем занимается, что знает и "
        "выбрал — это longterm с видом profile, а не этот слой. Формат слоя "
        "profile другой: объект "
        "«поле → новое значение» без value и kind, только изменённые поля. "
        "Поля и допустимые значения: " + "; ".join(fields) + ". "
        "Просьбу, которую выражает одно из полей выше (язык, тон, объём, форма, "
        "длина, код), записывай только в это поле и не добавляй её в "
        "constraints; constraints — только для правил, которых нет среди полей, "
        "коротко и своими словами, без копирования реплики. "
        'Примеры: «впредь отвечай по-английски» → {"profile": {"language": "en"}} '
        "и больше ничего; "
        
        '«впредь без платных сервисов» → {"profile": {"constraints": '
        '[...прежние, "не предлагать платные сервисы"]}}.'
    )


def _profile_changes(
    profile: UserProfile | None, changes: dict
) -> tuple[dict[str, dict], list[str]]:
    """Keep the fields the schema accepts; name the rest as refused.

    Each field is checked on a whole new profile, so a value is judged by the
    same model the form uses. The candidate stores the normalized value as
    JSON: it is what accepting it will write.
    """
    if profile is None:
        return {}, [f"profile.{field}" for field in changes]
    accepted: dict[str, dict] = {}
    refused: list[str] = []
    current = profile.model_dump(mode="json")
    for field, raw in changes.items():
        value = raw.get("value") if isinstance(raw, dict) and "value" in raw else raw
        if field not in ROUTER_PROFILE_FIELDS:
            refused.append(f"profile.{field}")
            continue
        try:
            candidate = UserProfile.model_validate({**current, field: value})
        except ValidationError:
            refused.append(f"profile.{field}")
            continue
        new = candidate.model_dump(mode="json")[field]
        if new != current[field]:
            accepted[field] = {
                "value": json.dumps(new, ensure_ascii=False),
                "kind": PROFILE_PROPOSAL_KIND,
            }
    return accepted, refused


def _summary_text(summary: dict) -> str:
    return str(summary.get("text") or summary.get("summary") or "")


def _with_payload(
    layer: dict | None, tokens: int, dropped: list[str] | None = None
) -> dict | None:
    if not layer:
        return None
    left_out = {"dropped_keys": list(dropped)} if dropped is not None else {}
    return {**layer, "payload_tokens": tokens, **left_out}


def _merge_layer(
    previous: dict[str, dict],
    changes: dict[str, dict],
    seq: int,
    layer: str,
) -> tuple[dict[str, dict], dict[str, dict], list[str]]:
    """Apply one layer's changes on top of its stored memory.

    Small models answer with only the keys the new turn touched, so replacing
    the memory with their object would silently forget earlier items; an
    explicit empty value is the only way a key leaves the memory.  A kind the
    router names has to belong to this layer, otherwise the item is refused
    instead of joining the layer under a foreign kind; a key that was already
    there keeps the kind it was routed to, because a rephrased value is not a
    reroute.
    """
    merged = dict(previous)
    accepted: dict[str, dict] = {}
    refused: list[str] = []
    for key, change in changes.items():
        value = str(change.get("value") or "")
        kind = str(change.get("kind") or previous.get(key, {}).get("kind") or "")
        if not value:
            if key not in merged:
                continue
            accepted[key] = {"value": "", "kind": kind}
            merged.pop(key)
            continue
        if kind not in LAYER_KINDS[layer]:
            refused.append(f"{layer}.{key}")
            continue
        accepted[key] = {"value": value, "kind": kind}
        merged[key] = {"value": value, "kind": kind, "origin": "router", "seq": seq}
    # `accepted` is what a candidate list offers a human, so it carries the
    # resolved kind: nothing reaches a layer under a kind that was refused.
    return merged, accepted, refused


def _parse_layers(
    raw: str, layers: tuple[str, ...]
) -> tuple[dict[str, dict[str, dict]], list[str]]:
    """Decode the router answer into `layer -> key -> {value, kind}`."""
    text = raw.strip()
    if text.startswith("```json") and text.endswith("```"):
        text = text[7:-3].strip()
    elif text.startswith("```") and text.endswith("```"):
        text = text[3:-3].strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        raise LLMError(f"модель вернула память не в JSON: {error.msg}") from error
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise LLMError("модель вернула память не как JSON-объект")
    # A profile answer is `field -> value` with typed values, so it is kept
    # raw for the schema to judge instead of being flattened to strings.
    routed = {
        layer: value[layer] if layer == "profile" else _parse_changes(value[layer])
        for layer in layers
        if isinstance(value.get(layer), dict)
    }
    if not routed:
        raise LLMError(
            "модель не назвала слой памяти: ожидались ключи "
            + ", ".join(layers)
        )
    rejected = [key for key in value if key not in routed]
    return routed, rejected


def _parse_changes(layer: object) -> dict[str, dict]:
    """One layer's object, tolerating a bare value where a kind was asked."""
    if not isinstance(layer, dict):
        return {}
    changes: dict[str, dict] = {}
    for key, item in layer.items():
        if not isinstance(key, str):
            continue
        if isinstance(item, dict) and ("value" in item or "kind" in item):
            changes[key] = {
                "value": _value_text(item.get("value")),
                "kind": str(item.get("kind") or ""),
            }
        else:
            changes[key] = {"value": _value_text(item), "kind": ""}
    return changes


def _value_text(item: object) -> str:
    """Render one fact value as the string the memory stores.

    The prompt asks for strings, but models still answer with numbers, flags
    and lists; those carry the fact just as well once flattened.
    """
    if item is None:
        return ""
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, bool):
        return "да" if item else "нет"
    if isinstance(item, (int, float)):
        return str(item)
    if isinstance(item, list):
        return ", ".join(part for part in (_value_text(entry) for entry in item) if part)
    return json.dumps(item, ensure_ascii=False, sort_keys=True)


def _empty_helper_error(what: str, usage: Usage, budget: int) -> str:
    if usage.reasoning or usage.tokens_out >= budget:
        return (
            f"модель вернула {what}: бюджет {budget} токенов и запас "
            f"{HELPER_REASONING_HEADROOM} ушли на reasoning ({usage.reasoning}); "
            "увеличьте лимит или выберите модель без рассуждений"
        )
    return f"модель вернула {what}"
