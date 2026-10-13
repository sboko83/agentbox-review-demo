"""Phase-4 judge, quote and matrix contracts without paid calls."""

import json

import pytest

from agentbox.core.experiments import Price, estimate_matrix, execute_matrix
from agentbox.core.invariants import InvariantRule, InvariantSet
from agentbox.core.spec import AgentSpec
from agentbox.llm.stub import StubLLM
from agentbox.store import repository as repo
from agentbox.store.db import init_db


PRICES = {
    ("zen", "gpt-5-nano"): Price(0.05, 0.4),
    ("zen", "gpt-5.6-luna"): Price(0.2, 1.2),
}


def test_quote_includes_policy_call_count_judge_and_recovery_bound():
    spec = AgentSpec(
        strategy="experts",
        judge_failure_action="fallback",
        judge_fallback_model="gpt-5.6-luna",
    )

    quote = estimate_matrix("проверь ответ", 2, [spec], PRICES)

    # Four policy calls twice for the original and fallback answer, per repeat.
    assert quote.agent_calls == 16
    assert quote.judge_calls == 4
    assert quote.total_usd > 0
    assert quote.by_config[0]["judge_model"] == "gpt-5-nano"


def test_quote_keys_prices_by_provider_and_model():
    specs = [
        AgentSpec(
            model={"provider": "alpha", "model": "shared"},
            judge={"model": "shared"},
        ),
        AgentSpec(
            model={"provider": "beta", "model": "shared"},
            judge={"model": "shared"},
        ),
    ]

    quote = estimate_matrix(
        "проверь",
        1,
        specs,
        {
            ("alpha", "shared"): Price(1, 1),
            ("beta", "shared"): Price(10, 10),
        },
    )

    assert [item["provider"] for item in quote.by_config] == ["alpha", "beta"]
    assert quote.by_config[0]["maximum_usd"] < quote.by_config[1]["maximum_usd"]


@pytest.mark.asyncio
async def test_failed_judge_replaces_answer_with_fallback_and_report_uses_actual_cost(tmp_path):
    conn = init_db(tmp_path / "agentbox.db")
    spec = AgentSpec(
        name="fallback",
        judge_failure_action="fallback",
        judge_fallback_model="gpt-5.6-luna",
        judge_threshold=7,
    )
    run_id = repo.create_run(
        conn,
        name="test",
        prompt="задача",
        repeats=1,
        budget_usd=None,
        estimated_cost_usd=0.1,
        specs=[spec],
    )
    llm = StubLLM(
        script=[
            ("первый ответ", []),
            (json.dumps({"score": 2, "rationale": "ошибка"}), []),
            ("запасной ответ", []),
            (json.dumps({"score": 9, "rationale": "верно"}), []),
        ],
        cost=0.00001,
    )

    report = await execute_matrix(
        conn, llm, run_id=run_id, prompt="задача", specs=[spec], repeats=1
    )

    item = report["results"][0]
    assert item["answer"].strip() == "запасной ответ"
    assert item["resolution"] == "fallback"
    assert item["judge"]["passed"] is True
    assert len(item["attempts"]) == 2
    assert report["comparison"] == [
        {
            "config_index": 0,
            "name": "Конфигурация 1",
            "user_profile": None,
            "provider": "zen",
            "model": "gpt-5-nano",
            "runs": 1,
            "passed": 1,
            "average_score": 9.0,
            "actual_cost_usd": 0.00004,
        }
    ]
    assert report["totals"]["requests"] == 4
    assert report["totals"]["cost_usd"] == pytest.approx(0.00004)


@pytest.mark.asyncio
async def test_matrix_answer_enforces_the_quoted_invariant_snapshot(tmp_path):
    conn = init_db(tmp_path / "invariants.db")
    spec = AgentSpec()
    invariants = InvariantSet(
        rules=[
            InvariantRule(
                id="stack.python", category="stack", text="Использовать Python 3.9+."
            )
        ]
    )
    run_id = repo.create_run(
        conn,
        name="rules",
        prompt="пример",
        repeats=1,
        budget_usd=None,
        estimated_cost_usd=0.1,
        specs=[spec],
        invariant_sets=[invariants],
    )
    allow = json.dumps(
        {
            "decision": "allow",
            "rule_ids": [],
            "explanation": "конфликта нет",
            "alternative": None,
        },
        ensure_ascii=False,
    )
    llm = StubLLM(
        script=[
            (allow, []),
            ("ответ", []),
            (allow, []),
            (json.dumps({"score": 9, "rationale": "верно"}), []),
        ],
        cost=0.00001,
    )

    report = await execute_matrix(
        conn,
        llm,
        run_id=run_id,
        prompt="пример",
        specs=[spec],
        repeats=1,
        invariant_sets=[invariants],
    )

    assert "[stack.python]" in llm.requests[1].messages[1]["content"]
    checks = report["results"][0]["attempts"][0]["invariant_checks"]
    assert [item["outcome"] for item in checks] == ["allow", "allow"]
    assert report["totals"]["requests"] == 4
    assert report["results"][0]["actual_cost_usd"] == pytest.approx(0.00004)

    quote = estimate_matrix(
        "пример", 1, [spec], PRICES, invariant_sets=[invariants]
    )
    assert quote.agent_calls == 5  # answer, possible refusal, and three checks
