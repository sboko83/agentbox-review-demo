"""Judge-backed matrix experiments, kept outside the interactive chat path.

An experiment owns no agent registry and no provider semaphore: those are phase
5 concerns. Phase 4 executes one confirmed, immutable matrix and persists the
answers, judge decisions and measured spend needed for a comparison report.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from statistics import mean
from typing import Callable

from agentbox.core.agent import Agent
from agentbox.core.events import Done, Failed, InvariantChecked
from agentbox.core.invariants import InvariantSet
from agentbox.core.ledger import BudgetExceeded
from agentbox.core.ports import Chunk, LLMError, LLMPort, LLMRequest, Usage
from agentbox.core.spec import AgentSpec
from agentbox.core.user_profile import UserProfile
from agentbox.store import repository as repo


JUDGE_FORMAT = {
    "type": "json_schema",
    "name": "judge_verdict",
    "schema": {
        "type": "object",
        "properties": {
            "score": {"type": "integer", "minimum": 0, "maximum": 10},
            "rationale": {"type": "string"},
        },
        "required": ["score", "rationale"],
        "additionalProperties": False,
    },
}


# A reasoning model spends its output budget on thinking first: at 200 tokens
# gpt-5-nano returned an empty verdict. The cap bounds the quote, not the reply.
JUDGE_MAX_OUTPUT_TOKENS = 2000


class EstimateError(ValueError):
    """A run cannot be safely quoted from the current catalog."""


def responses_transport(_model: str) -> str:
    """Default for direct core tests whose stub models use Responses history."""
    return "responses"


@dataclass(frozen=True, slots=True)
class Price:
    """Published list prices per million tokens for one supported model."""

    input_usd_per_million: float
    output_usd_per_million: float


@dataclass(frozen=True, slots=True)
class CostQuote:
    """Conservative maximum before actual usage and provider cost are known."""

    total_usd: float
    agent_calls: int
    judge_calls: int
    by_config: list[dict]

    def as_dict(self) -> dict:
        return asdict(self)


def estimate_matrix(
    prompt: str,
    repeats: int,
    specs: list[AgentSpec],
    prices: dict[tuple[str, str], Price],
    config_names: list[str] | None = None,
    user_profiles: list[UserProfile | None] | None = None,
    invariant_sets: list[InvariantSet | None] | None = None,
) -> CostQuote:
    """Quote the worst case including one configured recovery per failed answer.

    List prices cannot predict reasoning tokens or an early successful judge,
    so this intentionally uses the configured output cap and the failure path.
    The confirmation screen calls it an upper estimate, while the report uses
    the provider's actual ``cost`` events.
    """
    if not prompt.strip():
        raise EstimateError("Запрос для прогона пуст")
    if not specs:
        raise EstimateError("Нужна хотя бы одна конфигурация")
    if repeats < 1 or repeats > 20:
        raise EstimateError("Повторов должно быть от 1 до 20")

    total = 0.0
    agent_calls = 0
    judge_calls = 0
    by_config: list[dict] = []
    for index, spec in enumerate(specs):
        target_price = _price_for(spec.model.provider, spec.model.model, prices)
        judge_price = _price_for(spec.model.provider, spec.judge.model, prices)
        fallback_price = (
            _price_for(spec.model.provider, spec.judge.fallback_model, prices)
            if spec.judge.failure_action == "fallback"
            else None
        )
        attempts = 1 if spec.judge.failure_action == "as_is" else 2
        target_calls = _strategy_calls(spec) * repeats
        retry_calls = _strategy_calls(spec) * repeats if attempts == 2 else 0
        answer_tokens = max(1, spec.output.max_output_tokens)
        profile = user_profiles[index] if user_profiles else None
        invariants = invariant_sets[index] if invariant_sets else None
        input_tokens = _estimate_input_tokens(prompt, spec, profile)
        base_cost = _call_cost(target_price, input_tokens, answer_tokens) * target_calls
        recovery_cost = 0.0
        if retry_calls:
            recovery_cost = _call_cost(
                fallback_price or target_price, input_tokens, answer_tokens
            ) * retry_calls
        # A judge sees the question, criteria and a full capped answer.
        judge_input = _judge_input_tokens(prompt, spec, answer_tokens)
        judge_cost = _call_cost(judge_price, judge_input, JUDGE_MAX_OUTPUT_TOKENS) * repeats * attempts
        guard_calls = 0
        guard_cost = 0.0
        refusal_calls = 0
        refusal_cost = 0.0
        if invariants and invariants.rules:
            guard_price = _price_for(spec.model.provider, spec.task.model, prices)
            block_tokens = max(1, (len(invariants.render()) + len(prompt)) // 3)
            guard_output = max(spec.task.max_tokens, 300) + 2048
            # Worst case: the request passes, the first answer conflicts, and
            # a format-preserving refusal is generated and checked once more.
            guard_calls = 3 * repeats * attempts
            guard_cost = (
                _call_cost(guard_price, block_tokens + 200, guard_output)
                + 2 * _call_cost(
                    guard_price,
                    block_tokens + answer_tokens + 200,
                    guard_output,
                )
            ) * repeats * attempts
            refusal_calls = repeats * attempts
            refusal_cost = _call_cost(
                target_price, input_tokens, answer_tokens
            ) * repeats
            if attempts == 2:
                refusal_cost += _call_cost(
                    fallback_price or target_price, input_tokens, answer_tokens
                ) * repeats
        subtotal = base_cost + recovery_cost + judge_cost + guard_cost + refusal_cost
        total += subtotal
        agent_calls += target_calls + retry_calls + guard_calls + refusal_calls
        judge_calls += repeats * attempts
        by_config.append(
            {
                "index": index,
                "name": config_names[index] if config_names else f"Конфигурация {index + 1}",
                "provider": spec.model.provider,
                "model": spec.model.model,
                "judge_model": spec.judge.model,
                "maximum_usd": round(subtotal, 6),
            }
        )
    return CostQuote(round(total, 6), agent_calls, judge_calls, by_config)


def _price_for(
    provider: str, model: str | None, prices: dict[tuple[str, str], Price]
) -> Price:
    if not model:
        raise EstimateError("Для fallback нужна запасная модель")
    price = prices.get((provider, model))
    if price is None:
        raise EstimateError(f"Нет полной цены модели {provider}/{model} в каталоге")
    return price


def _strategy_calls(spec: AgentSpec) -> int:
    return {"direct": 1, "step_by_step": 1, "self_prompt": 2, "experts": 4}[
        spec.reasoning.strategy
    ]


def _estimate_input_tokens(
    prompt: str, spec: AgentSpec, profile: UserProfile | None = None
) -> int:
    profile_chars = len(profile.render()) if profile else 0
    return max(1, (len(prompt) + len(spec.model.role) + profile_chars) // 3)


def _judge_input_tokens(prompt: str, spec: AgentSpec, answer_tokens: int) -> int:
    return max(1, (len(prompt) + len(spec.judge.criteria)) // 3) + answer_tokens


def _call_cost(price: Price, input_tokens: int, output_tokens: int) -> float:
    return (
        input_tokens * price.input_usd_per_million + output_tokens * price.output_usd_per_million
    ) / 1_000_000


async def execute_matrix(
    conn,
    llm: LLMPort,
    *,
    run_id: str,
    prompt: str,
    specs: list[AgentSpec],
    repeats: int,
    transport_for: Callable[[str], str] = responses_transport,
    user_profiles: list[UserProfile | None] | None = None,
    invariant_sets: list[InvariantSet | None] | None = None,
) -> dict:
    """Run a confirmed matrix and return the persisted comparison report."""
    try:
        for config_index, spec in enumerate(specs):
            for repeat_index in range(repeats):
                result = await execute_cell(
                    llm,
                    conn,
                    run_id,
                    prompt,
                    spec,
                    transport_for=transport_for,
                    user_profile=user_profiles[config_index] if user_profiles else None,
                    invariants=invariant_sets[config_index] if invariant_sets else None,
                )
                repo.record_run_result(
                    conn,
                    run_id,
                    config_index,
                    repeat_index,
                    result["status"],
                    result,
                )
    except Exception:
        repo.finish_run(conn, run_id, "failed")
        raise
    repo.finish_run(conn, run_id, "completed")
    report = repo.get_run(conn, run_id)
    assert report is not None
    return comparison_report(report)


async def execute_cell(
    llm: LLMPort,
    conn,
    run_id: str,
    prompt: str,
    spec: AgentSpec,
    *,
    transport_for: Callable[[str], str] = responses_transport,
    user_profile: UserProfile | None = None,
    invariant_hash: str | None = None,
    invariants: InvariantSet | None = None,
) -> dict:
    """One matrix cell: answer, judge verdict and the configured recovery.

    Public because the phase-5 supervisor runs the same cell concurrently; the
    only difference there is the wrapped, gated `llm` it passes in. The profile
    shapes only the answer: the judge scores it against the task criteria.
    """
    profile_hash = user_profile.hash if user_profile else None
    attempts: list[dict] = []
    invariant_hash = invariants.hash if invariants is not None else invariant_hash
    first = await _answer(llm, prompt, spec, transport_for, user_profile, invariants)
    _record_invariant_checks(conn, run_id, spec, first, invariant_hash)
    _record_usage(
        conn, run_id, spec, spec.model.model, first, profile_hash, invariant_hash
    )
    first["judge"] = await _judge(llm, prompt, first["answer"], spec)
    _record_usage(
        conn, run_id, spec, spec.judge.model, first["judge"], invariant_hash=invariant_hash
    )
    attempts.append(first)

    final = first
    resolution = "passed" if first["judge"]["passed"] else "as_is"
    if first["status"] == "ok" and not first["judge"]["passed"]:
        if spec.judge.failure_action == "retry":
            final = await _answer(
                llm, prompt, spec, transport_for, user_profile, invariants
            )
            _record_invariant_checks(conn, run_id, spec, final, invariant_hash)
            _record_usage(
                conn, run_id, spec, spec.model.model, final, profile_hash, invariant_hash
            )
            final["judge"] = await _judge(llm, prompt, final["answer"], spec)
            _record_usage(
                conn,
                run_id,
                spec,
                spec.judge.model,
                final["judge"],
                invariant_hash=invariant_hash,
            )
            attempts.append(final)
            resolution = "retry"
        elif spec.judge.failure_action == "fallback":
            fallback_spec = spec.model_copy(
                update={
                    "model": spec.model.model_copy(update={"model": spec.judge.fallback_model})
                }
            )
            final = await _answer(
                llm, prompt, fallback_spec, transport_for, user_profile, invariants
            )
            _record_invariant_checks(conn, run_id, fallback_spec, final, invariant_hash)
            _record_usage(
                conn,
                run_id,
                fallback_spec,
                fallback_spec.model.model,
                final,
                profile_hash,
                invariant_hash,
            )
            final["judge"] = await _judge(llm, prompt, final["answer"], spec)
            _record_usage(
                conn,
                run_id,
                spec,
                spec.judge.model,
                final["judge"],
                invariant_hash=invariant_hash,
            )
            attempts.append(final)
            resolution = "fallback"

    return {
        "status": final["status"],
        "answer": final["answer"],
        "judge": final["judge"],
        "resolution": resolution,
        "attempts": attempts,
        "actual_cost_usd": round(sum(_attempt_cost(attempt) for attempt in attempts), 6),
    }


async def _answer(
    llm: LLMPort,
    prompt: str,
    spec: AgentSpec,
    transport_for: Callable[[str], str],
    user_profile: UserProfile | None = None,
    invariants: InvariantSet | None = None,
) -> dict:
    agent = Agent(
        spec,
        llm,
        message_format=transport_for(spec.model.model),
        user_profile=user_profile,
        invariants=invariants,
    )
    started_at = time.perf_counter()
    answer = ""
    failure: str | None = None
    usage = Usage()
    checks: list[dict] = []
    async for event in agent.stream(prompt):
        if isinstance(event, Done):
            answer = event.text
            usage = Usage(
                tokens_in=int(event.usage.get("tokens_in", 0)),
                tokens_out=int(event.usage.get("tokens_out", 0)),
                reasoning=int(event.usage.get("reasoning", 0)),
                cost=float(event.usage.get("cost_usd", 0)),
            )
        elif isinstance(event, Failed):
            failure = event.message
        elif isinstance(event, InvariantChecked):
            checks.append(asdict(event))
    return {
        "status": "error" if failure else "ok",
        "answer": answer,
        "error": failure,
        "model": spec.model.model,
        "usage": asdict(usage),
        "invariant_checks": checks,
        "latency_ms": round((time.perf_counter() - started_at) * 1000),
    }


async def _judge(llm: LLMPort, prompt: str, answer: str, spec: AgentSpec) -> dict:
    started_at = time.perf_counter()
    request = LLMRequest(
        model=spec.judge.model,
        temperature=0,
        max_output_tokens=JUDGE_MAX_OUTPUT_TOKENS,
        reasoning_effort="low",
        text_format=JUDGE_FORMAT,
        messages=[
            {
                "role": "system",
                "content": "Ты независимый строгий судья. Оцени только JSON по заданной схеме.",
            },
            {
                "role": "user",
                "content": (
                    f"ЗАДАЧА:\n{prompt}\n\nКРИТЕРИИ:\n{spec.judge.criteria}\n\n"
                    f"ОТВЕТ ДЛЯ ОЦЕНКИ:\n{answer}"
                ),
            },
        ],
    )
    text, usage, error = await _collect(llm, request)
    if error:
        return {
            "available": False,
            "passed": False,
            "error": error,
            "usage": asdict(usage),
            "latency_ms": round((time.perf_counter() - started_at) * 1000),
        }
    try:
        payload = json.loads(text)
        score = int(payload["score"])
        if not 0 <= score <= 10:
            raise ValueError("балл вне диапазона")
        rationale = str(payload["rationale"])
    except (ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
        return {
            "available": False,
            "passed": False,
            "error": f"судья вернул невалидный JSON: {error}",
            "usage": asdict(usage),
            "latency_ms": round((time.perf_counter() - started_at) * 1000),
        }
    return {
        "available": True,
        "score": score,
        "rationale": rationale,
        "passed": score >= spec.judge.threshold,
        "usage": asdict(usage),
        "latency_ms": round((time.perf_counter() - started_at) * 1000),
    }


async def _collect(llm: LLMPort, request: LLMRequest) -> tuple[str, Usage, str | None]:
    parts: list[str] = []
    usage = Usage()
    try:
        async for chunk in llm.stream(request):
            if chunk.text:
                parts.append(chunk.text)
            if chunk.usage is not None:
                usage = chunk.usage
    except (BudgetExceeded, LLMError) as error:
        return "".join(parts), usage, str(error)
    return "".join(parts), usage, None


def _record_usage(
    conn,
    run_id: str,
    spec: AgentSpec,
    model: str,
    result: dict,
    user_profile_hash: str | None = None,
    invariant_hash: str | None = None,
) -> None:
    """Judge rows carry no hash: the judge never sees the profile."""
    repo.record_run_usage(
        conn,
        run_id,
        spec,
        model,
        Usage(**result["usage"]),
        latency_ms=result["latency_ms"],
        user_profile_hash=user_profile_hash,
        invariant_hash=invariant_hash,
    )


def _record_invariant_checks(
    conn,
    run_id: str,
    spec: AgentSpec,
    result: dict,
    invariant_hash: str | None,
) -> None:
    for checked in result.get("invariant_checks", []):
        usage = checked["usage"]
        repo.record_run_usage(
            conn,
            run_id,
            spec,
            checked["model"],
            Usage(
                tokens_in=int(usage.get("tokens_in", 0)),
                tokens_out=int(usage.get("tokens_out", 0)),
                reasoning=int(usage.get("reasoning", 0)),
                cost=float(usage.get("cost_usd", 0)),
            ),
            latency_ms=int(checked.get("latency_ms", 0)),
            invariant_hash=invariant_hash,
            kind="invariant",
        )


def _attempt_cost(attempt: dict) -> float:
    checks = sum(
        float(item["usage"].get("cost_usd", 0))
        for item in attempt.get("invariant_checks", [])
    )
    return (
        float(attempt["usage"]["cost"])
        + float(attempt["judge"]["usage"]["cost"])
        + checks
    )


def comparison_report(run: dict) -> dict:
    """Aggregate persisted cells without hiding failed or unjudgeable answers."""
    groups = []
    for config in run["configs"]:
        items = [item for item in run["results"] if item["config_index"] == config["index"]]
        scores = [item["judge"]["score"] for item in items if item["judge"].get("available")]
        passed = sum(bool(item["judge"].get("passed")) for item in items)
        groups.append(
            {
                "config_index": config["index"],
                "name": config["name"],
                "user_profile": config.get("user_profile"),
                "provider": config["spec"]["model"]["provider"],
                "model": config["spec"]["model"]["model"],
                "runs": len(items),
                "passed": passed,
                "average_score": round(mean(scores), 2) if scores else None,
                "actual_cost_usd": round(sum(item["actual_cost_usd"] for item in items), 6),
            }
        )
    return {**run, "comparison": groups}
