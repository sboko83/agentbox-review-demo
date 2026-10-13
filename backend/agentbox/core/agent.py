"""The agent: role, memory, spend, tools and the answer pipeline.

Performs no I/O of its own — the model arrives as an `LLMPort`, the tools as a
`ToolExecutor`. That is what makes an agent an ordinary object: creating a
hundred of them with different specs costs a hundred allocations, not a hundred
processes.

Pipeline: InputPolicy -> model -> tool-loop -> OutputPolicy. The policy
contracts are injected just like the ports, so adding a plugin does not add a
branch to this agent.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import time
from typing import AsyncIterator

from agentbox.core.context import ContextState, ContextStrategy, ContextView, strategy_for
from agentbox.core.events import (
    AgentEvent,
    ApprovalNeeded,
    ContextPrepared,
    Delta,
    Done,
    Failed,
    InvariantChecked,
    ToolFinished,
    ToolStarted,
    UsageReported,
)
from agentbox.core.invariants import CheckTarget, InvariantDecision, InvariantGuard, InvariantSet
from agentbox.core.ledger import BudgetExceeded, Ledger
from agentbox.core.memory import Conversation, Message
from agentbox.core.policies import (
    InputPolicy,
    InputPolicyContext,
    OutputPolicy,
    OutputPolicyContext,
    default_input_policy,
    default_output_policy,
)
from agentbox.core.ports import (
    ApprovalPort,
    LLMError,
    LLMPort,
    LLMRequest,
    ToolCall,
    ToolExecutor,
    ToolResult,
    ToolSpec,
    Usage,
)
from agentbox.core.spec import AgentSpec
from agentbox.core.task_machine import (
    PLANNING_ROLE,  # re-exported: the planner role belongs to the stage
    TaskState,
    permits_tool,
    stage_role,
)
from agentbox.core.user_profile import UserProfile

REFUSED = "пользователь отклонил вызов"
READ_ONLY = "сессия только для чтения, изменяющие инструменты недоступны"


class ContextOverflow(RuntimeError):
    """The estimated provider input does not fit the effective context window."""


@dataclass(slots=True)
class _Estimate:
    role: int
    history: int
    request: int
    chars: int

    @property
    def total(self) -> int:
        return self.role + self.history + self.request


@dataclass(slots=True)
class _TurnAccounting:
    calls: int = 0
    role: int = 0
    history: int = 0
    request: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float = 0.0
    last_estimate: int = 0
    last_input_tokens: int = 0
    last_output_tokens: int = 0
    latency_ms: int = 0

    def add(self, estimate: _Estimate, usage: Usage, latency_ms: int) -> None:
        self.calls += 1
        self.role += estimate.role
        self.history += estimate.history
        self.request += estimate.request
        self.input_tokens += usage.tokens_in
        self.output_tokens += usage.tokens_out
        self.reasoning_tokens += usage.reasoning
        self.cost_usd += usage.cost
        self.last_estimate = estimate.total
        self.last_input_tokens = usage.tokens_in
        self.last_output_tokens = usage.tokens_out
        self.latency_ms += latency_ms


class Agent:
    """One configured agent with its own memory, spend and workspace."""

    def __init__(
        self,
        spec: AgentSpec,
        llm: LLMPort,
        executor: ToolExecutor | None = None,
        approvals: ApprovalPort | None = None,
        input_policy: InputPolicy | None = None,
        output_policy: OutputPolicy | None = None,
        context_strategy: ContextStrategy | None = None,
        task: TaskState | None = None,
        model_context_window: int | None = None,
        chars_per_token: float | None = None,
        message_format: str = "responses",
        user_profile: UserProfile | None = None,
        invariants: InvariantSet | None = None,
    ) -> None:
        self.spec = spec
        # Read once per turn by whoever builds the agent: the version sent in
        # this turn's payload is the one its ledger rows cite.
        self.user_profile = user_profile
        self.invariants = invariants or InvariantSet()
        self._llm = llm
        self._executor = executor
        # The stage decides the role suffix and the tool list; the row in
        # the database is the authority, never anything the chat says.
        self.task = task
        self._approvals = approvals
        self._input_policy = input_policy or default_input_policy(spec)
        self._output_policy = output_policy or default_output_policy(spec)
        self._context_strategy = context_strategy or strategy_for(spec.context)
        self._context_limit = spec.context.window_tokens or model_context_window
        self._context_source = "spec" if spec.context.window_tokens is not None else "model"
        self._message_format = message_format
        suffix = stage_role(task)
        role = f"{spec.model.role}\n\n{suffix}" if suffix else spec.model.role
        self.conversation = Conversation(role=role, limit_tokens=2_000_000)
        if chars_per_token is not None and chars_per_token > 0:
            self.conversation.chars_per_token = chars_per_token
        self.ledger = Ledger(budget_usd=spec.context.budget_usd)
        self._turn = _TurnAccounting()
        self._summary: dict | None = None
        self._working: dict | None = None
        self._longterm: dict | None = None
        self._last_context: ContextView | None = None
        self._guard = InvariantGuard(spec.task.model, max(spec.task.max_tokens, 300))

    def load_history(self, entries: list[dict]) -> None:
        """Restore stored dialogue; trimming waits for the next user question."""
        for entry in entries:
            if entry["role"] == "tool":
                self.conversation.add_tool(
                    call_id=entry["call_id"],
                    tool=entry["tool"],
                    arguments_json=entry["arguments_json"],
                    output=entry["content"],
                    seq=entry.get("seq"),
                )
            else:
                self.conversation.add(entry["role"], entry["content"], seq=entry.get("seq"))

    def load_summary(self, summary: dict | str | None) -> None:
        """Restore the latest persisted summary before loading its live tail."""
        if isinstance(summary, str):
            self._summary = {"text": summary}
        else:
            self._summary = summary

    def load_memory(
        self, *, working: dict | None = None, longterm: dict | None = None
    ) -> None:
        """Attach the memory layers this spec enabled; the rest stay empty."""
        self._working = working
        self._longterm = longterm

    def context_view(self) -> ContextView:
        """Build the current provider view without mutating transcript history."""
        return self._context_strategy.build(
            self.conversation.messages,
            ContextState(
                role=self.conversation.role,
                chars_per_token=self.conversation.chars_per_token,
                summary=self._summary,
                working=self._working,
                longterm=self._longterm,
                working_max_tokens=self.spec.memory.working_max_tokens,
                longterm_max_tokens=self.spec.memory.longterm_max_tokens,
                invariants=self.invariants,
                user_profile=self.user_profile,
                task=self.task,
            ),
        )

    @property
    def user_profile_hash(self) -> str | None:
        return self.user_profile.hash if self.user_profile else None

    @property
    def tools(self) -> list[ToolSpec]:
        """What this agent may call, given its executor and approval mode."""
        if self._executor is None or not permits_tool(self.task, "", False):
            return []
        available = self._executor.describe()
        if self.spec.tools.approval_mode == "read-only":
            available = [tool for tool in available if not tool.mutating]
        return [
            tool for tool in available if permits_tool(self.task, tool.name, tool.mutating)
        ]

    async def stream(self, text: str) -> AsyncIterator[AgentEvent]:
        """Answer one message without exposing anything the invariant gate refused."""
        self._turn = _TurnAccounting()
        self.conversation.add("user", text)

        answer: list[str] = []
        turn = Usage()
        forced_prompt: str | None = None
        refusal_attempted = False
        guarded_context: ContextView | None = None
        if self.invariants.rules:
            try:
                # The rules must fit in full before even the checker is called;
                # a helper call is not a valid way to discover an impossible payload.
                _, _, guarded_context = self._prepare_call(text, [], True)
            except ContextOverflow as error:
                yield self._abort(str(error), answer)
                return
        try:
            request_check = await self._guard.check(
                self._llm, self.invariants, "request", text, self.ledger
            )
        except BudgetExceeded as error:
            yield self._check_event(self._guard_error("request", str(error)))
            yield self._abort(str(error), answer)
            return
        if request_check is not None:
            yield self._check_event(request_check)
            if request_check.outcome == "error":
                yield self._abort(
                    "Проверка инвариантов не выполнена: " + request_check.explanation,
                    answer,
                )
                return
            if request_check.outcome == "conflict":
                if self.spec.output.mode == "free":
                    refusal = self._safe_refusal(request_check)
                    if guarded_context is not None:
                        yield ContextPrepared(view=self._context_dict(guarded_context))
                    yield self._check_event(
                        InvariantDecision(
                            target="response",
                            outcome="allow",
                            rule_ids=request_check.rule_ids,
                            explanation=(
                                "Безопасный отказ сформирован детерминированно из "
                                "применённых правил; нарушающее решение не создавалось."
                            ),
                            alternative=(
                                "Сформулировать запрос в границах применённых правил."
                            ),
                        )
                    )
                    answer.append(refusal)
                    self.conversation.add("assistant", refusal)
                    yield Delta(text=refusal)
                    yield self._usage_report()
                    yield Done(
                        text=refusal,
                        usage={
                            "tokens_in": 0,
                            "tokens_out": 0,
                            "reasoning": 0,
                            "cost_usd": 0.0,
                        },
                        ledger=self.ledger.snapshot(),
                        memory_tokens=(
                            guarded_context.estimated_tokens if guarded_context else 0
                        ),
                        dropped=(
                            len(guarded_context.dropped_seqs) if guarded_context else 0
                        ),
                        steps=0,
                    )
                    return
                forced_prompt = self._refusal_prompt(request_check)
                refusal_attempted = True

        plan = self._input_policy.apply(
            InputPolicyContext(spec=self.spec, question=text, user_profile=self.user_profile)
        )
        prepared: dict[str, str] = {}

        # Self-prompt and expert strategies make their preparatory calls first.
        # They deliberately get no tools: only the final response may change a
        # coding workspace, and OutputPolicy still observes every requested call.
        for stage in [] if forced_prompt is not None else plan.stages:
            try:
                stage_parts: list[str] = []
                async for chunk in self._model_call(
                    plan.stage_prompt(stage, prepared),
                    [],
                    final=False,
                ):
                    if chunk.text:
                        stage_parts.append(chunk.text)
                    if chunk.usage is not None:
                        turn = _add(turn, chunk.usage)
            except (BudgetExceeded, ContextOverflow, LLMError) as error:
                if self._turn.calls:
                    yield self._usage_report()
                yield self._abort(str(error), answer)
                return
            prepared[stage.name] = "".join(stage_parts)

        tools = [] if forced_prompt is not None else self.tools
        step = 0

        # One extra step beyond the limit, called without tools, so the model
        # is forced to answer in words instead of the loop ending in silence.
        while step <= self.spec.tools.max_tool_steps:
            offered = tools if step < self.spec.tools.max_tool_steps else []
            prompt = forced_prompt or plan.final_prompt(text, prepared)
            try:
                parts: list[str] = []
                calls: list[ToolCall] = []
                request, estimate, context = self._prepare_call(prompt, offered, True)
                yield ContextPrepared(view=self._context_dict(context))
                async for chunk in self._model_call(
                    prompt, offered, request=request, estimate=estimate, context=context
                ):
                    if chunk.text:
                        parts.append(chunk.text)
                        if not self.invariants.rules:
                            yield Delta(text=chunk.text)
                    if chunk.tool_call is not None:
                        calls.append(chunk.tool_call)
                    if chunk.usage is not None:
                        turn = _add(turn, chunk.usage)
            except (BudgetExceeded, ContextOverflow, LLMError) as error:
                if self._turn.calls:
                    yield self._usage_report()
                yield self._abort(str(error), answer)
                return
            step_text = "".join(parts)

            subject = step_text
            if refusal_attempted:
                subject = (
                    "Контекст проверки: этот текст — кандидат на отказ после уже "
                    "найденного конфликта. Упоминание запрещённого варианта ради "
                    "объяснения не является предложением нарушить правило; ищи "
                    "только фактический код, команды, структуру или шаги запрещённой "
                    "реализации.\n\nТекст отказа:\n" + step_text
                )
            if calls:
                rendered = "\n".join(
                    f"tool={call.tool} arguments={call.arguments_json}" for call in calls
                )
                subject = f"{subject}\n\nЗапрошенные инструменты:\n{rendered}".strip()
            try:
                response_check = await self._guard.check(
                    self._llm, self.invariants, "response", subject, self.ledger
                )
            except BudgetExceeded as error:
                yield self._check_event(self._guard_error("response", str(error)))
                yield self._usage_report()
                yield self._abort(str(error), answer)
                return
            if response_check is not None:
                yield self._check_event(response_check)
                if response_check.outcome != "allow":
                    if response_check.outcome == "conflict" and not refusal_attempted:
                        forced_prompt = self._refusal_prompt(response_check)
                        refusal_attempted = True
                        tools = []
                        offered = []
                        continue
                    yield self._usage_report()
                    prefix = (
                        "Подготовленный ответ нарушает инварианты"
                        if response_check.outcome == "conflict"
                        else "Проверка подготовленного ответа не выполнена"
                    )
                    yield self._abort(f"{prefix}: {response_check.explanation}", answer)
                    return

            if step_text:
                answer.append(step_text)
                self.conversation.add("assistant", step_text)
                if self.invariants.rules:
                    yield Delta(text=step_text)

            if not calls:
                break

            # A provider must not be able to turn an empty tool list into a
            # local action by emitting an unsolicited function call.
            if not self.tools:
                break

            for call in calls:
                async for event in self._handle(call):
                    yield event
            step += 1

        yield self._usage_report()
        yield Done(
            text="".join(answer),
            usage={
                "tokens_in": turn.tokens_in,
                "tokens_out": turn.tokens_out,
                "reasoning": turn.reasoning,
                "cost_usd": round(turn.cost, 6),
            },
            ledger=self.ledger.snapshot(),
            memory_tokens=self._last_context.estimated_tokens if self._last_context else 0,
            dropped=len(self._last_context.dropped_seqs) if self._last_context else 0,
            steps=step + 1,
        )

    def _check_event(self, decision: InvariantDecision) -> InvariantChecked:
        return InvariantChecked(
            target=decision.target,
            outcome=decision.outcome,
            rule_ids=list(decision.rule_ids),
            explanation=decision.explanation,
            alternative=decision.alternative,
            model=self._guard.model,
            usage={
                "tokens_in": decision.usage.tokens_in,
                "tokens_out": decision.usage.tokens_out,
                "reasoning": decision.usage.reasoning,
                "cost_usd": decision.usage.cost,
            },
            input_estimate_tokens=decision.input_estimate_tokens,
            latency_ms=decision.latency_ms,
        )

    @staticmethod
    def _guard_error(target: CheckTarget, message: str) -> InvariantDecision:
        return InvariantDecision(
            target=target,
            outcome="error",
            explanation=message,
        )

    @staticmethod
    def _refusal_prompt(decision: InvariantDecision) -> str:
        ids = ", ".join(decision.rule_ids)
        alternative = (
            f" Допустимая альтернатива: {decision.alternative}."
            if decision.alternative
            else ""
        )
        return (
            "Ответь пользователю кратким отказом. Назови правила "
            f"[{ids}] и объясни: {decision.explanation}.{alternative} "
            "Не пересказывай и не выполняй запрещённое решение. Соблюди настроенный "
            "машинный формат ответа, если он задан."
        )

    def _safe_refusal(self, decision: InvariantDecision) -> str:
        by_id = {rule.id: rule for rule in self.invariants.rules}
        rules = [by_id[rule_id] for rule_id in decision.rule_ids]
        rendered = "\n".join(f"- [{rule.id}] {rule.text}" for rule in rules)
        return (
            "Не могу выполнить запрос: он конфликтует с обязательными правилами.\n"
            f"{rendered}\n"
            "Допустимая альтернатива: сформулировать решение, которое соблюдает "
            + ", ".join(f"[{rule.id}]" for rule in rules)
            + "."
        )

    def _usage_report(self) -> UsageReported:
        """Describe the completed provider calls in the current user turn."""
        estimate_total = self._turn.role + self._turn.history + self._turn.request
        return UsageReported(
            estimate={
                "role_tokens": self._turn.role,
                "history_tokens": self._turn.history,
                "request_tokens": self._turn.request,
                "input_tokens": estimate_total,
            },
            actual={
                "input_tokens": self._turn.input_tokens,
                "output_tokens": self._turn.output_tokens,
                "reasoning_tokens": self._turn.reasoning_tokens,
                "cost_usd": round(self._turn.cost_usd, 6),
                "estimate_delta_tokens": self._turn.input_tokens - estimate_total,
                "latency_ms": self._turn.latency_ms,
            },
            context={
                "estimated_tokens": self._turn.last_estimate,
                "input_tokens": self._turn.last_input_tokens,
                "output_tokens": self._turn.last_output_tokens,
                "limit_tokens": self._context_limit,
                "limit_source": self._context_source if self._context_limit is not None else None,
                "dropped_messages": len(self._last_context.dropped_seqs)
                if self._last_context
                else 0,
                "chars_per_token": round(self.conversation.chars_per_token, 4),
            },
            ledger=self.ledger.snapshot(),
        )

    def _history(self, context: ContextView | None = None) -> list[dict]:
        """The dialogue in the history format this model's endpoint expects."""
        view = context or self.context_view()
        return view.render(self._message_format)

    async def _model_call(
        self,
        prompt: str,
        tools: list[ToolSpec],
        final: bool = True,
        *,
        request: LLMRequest | None = None,
        estimate: _Estimate | None = None,
        context: ContextView | None = None,
    ) -> AsyncIterator:
        """Run one provider call and let OutputPolicy see its requested tools.

        `final` is false for a strategy's hidden preparatory call: the policy
        still observes it, but must not reshape an intermediate artefact.
        """
        self.ledger.check_budget()
        if request is None or estimate is None or context is None:
            request, estimate, context = self._prepare_call(prompt, tools, final)

        parts: list[str] = []
        calls: list[ToolCall] = []
        usage = Usage()
        started_at = time.perf_counter()
        async for chunk in self._llm.stream(request):
            if chunk.text:
                parts.append(chunk.text)
            if chunk.tool_call is not None:
                calls.append(chunk.tool_call)
            if chunk.usage is not None:
                usage = chunk.usage
            yield chunk
        self._output_policy.apply(
            OutputPolicyContext(
                spec=self.spec,
                phase="after_llm",
                request=request,
                text="".join(parts),
                tool_calls=calls,
                final=final,
            )
        )
        self.ledger.record(usage)
        self.conversation.calibrate(usage.tokens_in, estimate.chars)
        self._turn.add(estimate, usage, round((time.perf_counter() - started_at) * 1000))
        self._last_context = context

    def _request(
        self, prompt: str, tools: list[ToolSpec], final: bool, context: ContextView
    ) -> LLMRequest:
        """Build a provider request from the selected context view."""
        request = LLMRequest(
            model=self.spec.model.model,
            messages=self._replace_last_user(self._history(context), prompt),
            temperature=self.spec.sampling.temperature,
            tools=tools,
            top_p=self.spec.sampling.top_p,
            top_k=self.spec.sampling.top_k,
            frequency_penalty=self.spec.sampling.frequency_penalty,
            presence_penalty=self.spec.sampling.presence_penalty,
            stop=self.spec.sampling.stop_sequences(),
            seed=self.spec.sampling.seed,
            reasoning_effort=self.spec.model.reasoning_effort,
        )
        self._output_policy.apply(
            OutputPolicyContext(
                spec=self.spec, phase="before_llm", request=request, final=final
            )
        )
        return request

    def _prepare_call(
        self, prompt: str, tools: list[ToolSpec], final: bool
    ) -> tuple[LLMRequest, _Estimate, ContextView]:
        """Apply strategy first, then enforce the effective provider window."""
        context = self.context_view()
        request = self._request(prompt, tools, final, context)
        estimate = _estimate_request(
            request, self.conversation.chars_per_token, self._message_format
        )
        if (
            self.invariants.rules
            and self.spec.context.mode == "tokens"
            and estimate.total > self.spec.context.memory_limit_tokens
        ):
            raise ContextOverflow(
                "Контекст с полным блоком инвариантов не помещается в бюджет стратегии: "
                f"оценка {estimate.total} токенов превышает лимит "
                f"{self.spec.context.memory_limit_tokens}."
            )
        while self._context_limit is not None and estimate.total > self._context_limit:
            if self.spec.context.overflow == "send":
                if self.invariants.rules:
                    raise ContextOverflow(
                        "Контекст с полным блоком инвариантов не помещается: оценка "
                        f"{estimate.total} токенов превышает лимит {self._context_limit}."
                    )
                break
            if self.spec.context.overflow != "drop" or not context.drop_oldest():
                raise ContextOverflow(
                    "Контекст не помещается: оценка "
                    f"{estimate.total} токенов превышает лимит {self._context_limit} "
                    f"(режим {self.spec.context.overflow})."
                )
            request = self._request(prompt, tools, final, context)
            estimate = _estimate_request(
                request, self.conversation.chars_per_token, self._message_format
            )
        context.estimated_tokens = estimate.total
        self._last_context = context
        return request, estimate, context

    def _context_dict(self, context: ContextView) -> dict:
        return context.as_dict(
            limit_tokens=self._context_limit,
            limit_source=self._context_source,
            overflow=self.spec.context.overflow,
        )

    @staticmethod
    def _replace_last_user(messages: list[dict], prompt: str) -> list[dict]:
        """Keep stored history intact while a policy changes this request only."""
        result = [item.copy() for item in messages]
        for item in reversed(result):
            if item.get("role") == "user":
                item["content"] = prompt
                return result
        raise RuntimeError("в диалоге нет пользовательского сообщения")

    # --- tool loop --------------------------------------------------------

    def _mutating(self, name: str) -> bool:
        """Unknown names are not mutating: the executor refuses them anyway."""
        assert self._executor is not None
        return any(t.name == name and t.mutating for t in self._executor.describe())

    async def _confirm(self, call: ToolCall, preview: str) -> bool:
        """No one to ask means no. A silent yes is the one unacceptable answer."""
        if self._approvals is None:
            return False
        return await self._approvals.confirm(call, preview)

    async def _handle(self, call: ToolCall) -> AsyncIterator[AgentEvent]:
        """Approve, run and remember one tool call."""
        executor = self._executor
        assert executor is not None  # a call only exists if tools were offered
        preview = executor.preview(call)
        mode = self.spec.tools.approval_mode
        mutating = self._mutating(call.tool)

        blocked: str | None = None
        if mutating and self.invariants.rules:
            try:
                decision = await self._guard.check(
                    self._llm,
                    self.invariants,
                    "tool",
                    f"{preview}\narguments={call.arguments_json}",
                    self.ledger,
                )
            except BudgetExceeded as error:
                decision = self._guard_error("tool", str(error))
                yield self._check_event(decision)
                blocked = f"проверка инвариантов не выполнена: {error}"
            else:
                assert decision is not None
                yield self._check_event(decision)
                if decision.outcome != "allow":
                    rules = f" [{', '.join(decision.rule_ids)}]" if decision.rule_ids else ""
                    blocked = f"инварианты{rules}: {decision.explanation}"

        if blocked is not None:
            result = ToolResult(call_id=call.id, output=blocked, ok=False)
        elif mutating and mode == "read-only":
            result = ToolResult(call_id=call.id, output=READ_ONLY, ok=False)
        else:
            approved = True
            if mutating and mode == "ask":
                yield ApprovalNeeded(
                    call_id=call.id,
                    tool=call.tool,
                    arguments=call.arguments_json,
                    preview=preview,
                )
                approved = await self._confirm(call, preview)
            if approved:
                yield ToolStarted(
                    call_id=call.id,
                    tool=call.tool,
                    arguments=call.arguments_json,
                    preview=preview,
                )
                result = await executor.run(call)
            else:
                result = ToolResult(call_id=call.id, output=REFUSED, ok=False)

        self.conversation.add_tool(
            call_id=call.id,
            tool=call.tool,
            arguments_json=call.arguments_json,
            output=result.output,
        )
        yield ToolFinished(
            call_id=call.id,
            tool=call.tool,
            output=result.output,
            ok=result.ok,
            exit_code=result.exit_code,
            duration_ms=result.duration_ms,
        )

    def _abort(self, message: str, answer: list[str]) -> Failed:
        """Give up on this turn, leaving memory in a state a retry can reuse."""
        last = self.conversation.messages[-1] if self.conversation.messages else None
        if not answer and isinstance(last, Message) and last.role == "user":
            # Nothing was said, so the question stays unanswered in memory —
            # drop it, otherwise a retry sends the same text twice.
            self.conversation.messages.pop()
        return Failed(message=message)


def _add(total: Usage, part: Usage) -> Usage:
    """Turn totals: one turn can span several model calls."""
    return Usage(
        tokens_in=total.tokens_in + part.tokens_in,
        tokens_out=total.tokens_out + part.tokens_out,
        reasoning=total.reasoning + part.reasoning,
        cost=total.cost + part.cost,
    )


def _estimate_request(
    request: LLMRequest, chars_per_token: float, message_format: str
) -> _Estimate:
    """Estimate token-bearing payload parts and keep their UI attribution."""
    last_user = max(
        (index for index, item in enumerate(request.messages) if item.get("role") == "user"),
        default=-1,
    )
    buckets = {"role": 0, "history": 0, "request": 0}
    for index, item in enumerate(request.messages):
        chars = len(json.dumps(item, ensure_ascii=False, separators=(",", ":")))
        if item.get("role") == "system" and index == 0:
            buckets["role"] += chars
        elif item.get("role") == "system":
            buckets["history"] += chars
        elif index == last_user:
            buckets["request"] += chars
        else:
            buckets["history"] += chars

    tools = [
        tool.as_function() if message_format == "responses" else tool.as_chat_function()
        for tool in request.tools
    ]
    if tools:
        buckets["history"] += len(
            json.dumps(tools, ensure_ascii=False, separators=(",", ":"))
        )
    if request.text_format is not None:
        buckets["history"] += len(
            json.dumps(request.text_format, ensure_ascii=False, separators=(",", ":"))
        )

    def tokens(chars: int) -> int:
        return math.ceil(chars / chars_per_token) if chars else 0

    return _Estimate(
        role=tokens(buckets["role"]),
        history=tokens(buckets["history"]),
        request=tokens(buckets["request"]),
        chars=sum(buckets.values()),
    )
