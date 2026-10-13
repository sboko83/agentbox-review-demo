import math

import pytest

from agentbox.core.context import (
    HELPER_REASONING_HEADROOM,
    WORKING_HEADER,
    ContextState,
    MemoryRouter,
    NoneContextStrategy,
    SummaryContextStrategy,
    WindowContextStrategy,
    strategy_for,
)
from agentbox.core.ledger import Ledger
from agentbox.core.memory import Message
from agentbox.core.spec import AgentSpec
from agentbox.llm.stub import StubLLM


def router(layers: tuple[str, ...] = ("working",)) -> MemoryRouter:
    spec = AgentSpec(memory={"working_enabled": True, "model": "gpt-5-nano", "max_tokens": 64})
    return MemoryRouter(spec.memory.model, spec.memory.max_tokens, layers)


def item(value: str, kind: str = "step", seq: int = 2) -> dict:
    return {"value": value, "kind": kind, "origin": "router", "seq": seq}


def version(items: dict, covered_seq: int = 2) -> dict:
    return {"version": 1, "covered_seq": covered_seq, "items": items, "tokens": 4}


def history() -> list[dict]:
    return [
        {"seq": 1, "role": "user", "content": "Меня зовут Сергей."},
        {"seq": 2, "role": "assistant", "content": "Запомнил."},
        {"seq": 3, "role": "user", "content": "Я выбрал Kotlin."},
        {"seq": 4, "role": "assistant", "content": "Принято."},
    ]


def tail() -> list[Message]:
    return [
        Message("user", "старый", 1),
        Message("assistant", "ответ", 2),
        Message("user", "новый", 3),
        Message("assistant", "ответ", 4),
    ]


def test_layers_compose_under_any_strategy_in_a_stable_order():
    state = ContextState(
        role="роль",
        chars_per_token=3.0,
        working=version({"task": item("собрать ТЗ")}),
        longterm=version({"name": item("Сергей", kind="profile")}),
    )

    view = WindowContextStrategy(2).build(tail(), state)

    assert [entry.seq for entry in view.tail] == [3, 4]
    assert view.dropped_seqs == [1, 2]
    rendered = view.render("chat")
    # role, long-term, working, then the verbatim tail.
    assert [entry["role"] for entry in rendered] == [
        "system", "system", "system", "user", "assistant",
    ]
    assert "Долговременная память" in rendered[1]["content"]
    assert '"name": "Сергей"' in rendered[1]["content"]
    assert "Рабочая память" in rendered[2]["content"]
    assert '"task": "собрать ТЗ"' in rendered[2]["content"]

    # The strategy selects the tail and nothing else: the same state renders
    # the same layers when the whole transcript is kept.
    whole = NoneContextStrategy().build(tail(), state)
    assert len(whole.render("chat")) == 7


def test_layers_stay_empty_when_the_state_carries_none():
    state = ContextState(role="роль", chars_per_token=3.0)

    view = strategy_for(AgentSpec().context).build(tail(), state)

    assert view.working is None and view.longterm is None
    assert len(view.render("chat")) == 5


@pytest.mark.asyncio
async def test_router_names_the_layer_of_every_item_it_returns():
    llm = StubLLM(
        '{"working": {"goal": {"value": "собрать ТЗ", "kind": "goal"}},'
        ' "longterm": {"name": {"value": "Сергей", "kind": "profile"}}}'
    )

    result = await router(("working", "longterm")).extract(
        llm, history(), {"working": version({"task": item("старое", kind="step")})}, Ledger()
    )

    assert result is not None and result.error is None and result.updates is not None
    assert result.covered_seq == 4
    # The working layer keeps what it already held and gains the new goal.
    assert result.updates["working"].items == {
        "task": item("старое", kind="step"),
        "goal": {"value": "собрать ТЗ", "kind": "goal", "origin": "router", "seq": 4},
    }
    assert result.updates["longterm"].changes == {
        "name": {"value": "Сергей", "kind": "profile"}
    }
    role = llm.requests[0].messages[0]["content"]
    assert '{"working": {}, "longterm": {}}' in role
    assert "goal, constraint, step, artifact, blocker" in role
    assert "profile, decision, knowledge" in role
    assert llm.requests[0].model == "gpt-5-nano"
    assert llm.requests[0].max_output_tokens == 64 + HELPER_REASONING_HEADROOM
    assert llm.requests[0].text_format == {"type": "json_object"}


@pytest.mark.asyncio
async def test_router_is_only_told_about_the_layers_it_fills():
    llm = StubLLM('{"working": {"goal": {"value": "собрать ТЗ", "kind": "goal"}}}')

    await router(("working",)).extract(llm, history(), {}, Ledger())

    role = llm.requests[0].messages[0]["content"]
    assert "Слой working" in role and "Слой longterm" not in role
    assert "обязательный ключ верхнего уровня — working" in role
    assert '{"working": {}}' in role
    assert "Слой longterm" not in llm.requests[0].messages[1]["content"]


@pytest.mark.asyncio
async def test_router_refuses_a_kind_that_does_not_belong_to_the_layer():
    llm = StubLLM(
        '{"working": {"name": {"value": "Сергей", "kind": "profile"},'
        ' "goal": {"value": "собрать ТЗ", "kind": "goal"}}}'
    )

    result = await router(("working",)).extract(llm, history(), {}, Ledger())

    assert result is not None and result.updates is not None
    # The foreign kind is refused by name instead of joining the layer as a
    # plain item, and the rest of the answer still lands.
    assert result.rejected == ["working.name"]
    assert sorted(result.updates["working"].items) == ["goal"]


@pytest.mark.asyncio
async def test_router_keeps_the_kind_an_item_was_routed_to():
    llm = StubLLM('{"working": {"task": "собрать ТЗ до пятницы"}}')
    previous = {"working": version({"task": item("собрать ТЗ", kind="goal")})}

    result = await router(("working",)).extract(llm, history(), previous, Ledger())

    assert result is not None and result.updates is not None
    assert result.updates["working"].items["task"] == {
        "value": "собрать ТЗ до пятницы", "kind": "goal", "origin": "router", "seq": 4
    }


@pytest.mark.asyncio
async def test_router_refuses_a_new_key_that_names_no_kind():
    llm = StubLLM('{"working": {"task": "собрать ТЗ"}}')

    result = await router(("working",)).extract(llm, history(), {}, Ledger())

    assert result is not None and result.updates is not None
    assert result.rejected == ["working.task"]
    assert result.updates["working"].items == {}


@pytest.mark.asyncio
async def test_router_reads_only_what_the_watermark_has_not_covered():
    llm = StubLLM('{"working": {"goal": {"value": "собрать ТЗ", "kind": "goal"}}}')

    await router(("working",)).extract(llm, history(), {}, Ledger(), covered_seq=2)

    prompt = llm.requests[0].messages[1]["content"]
    assert "Меня зовут" not in prompt and "Я выбрал Kotlin" in prompt


@pytest.mark.asyncio
async def test_router_rejects_an_answer_that_names_no_layer():
    result = await router(("working",)).extract(
        StubLLM('{"name": "Сергей"}'), history(), {}, Ledger()
    )

    assert result is not None and result.updates is None
    assert "не назвала слой" in (result.error or "")


@pytest.mark.asyncio
async def test_router_rejects_non_json_memory():
    result = await router(("working",)).extract(
        StubLLM("не json"), history(), {}, Ledger()
    )

    assert result is not None
    assert result.updates is None
    assert "не в JSON" in (result.error or "")


@pytest.mark.asyncio
async def test_router_flattens_non_string_values():
    llm = StubLLM(
        '{"working": {"budget": {"value": 1.2, "kind": "constraint"},'
        ' "platforms": {"value": ["iOS", "Android"], "kind": "artifact"},'
        ' "cash": {"value": false, "kind": "constraint"}}}'
    )

    result = await router(("working",)).extract(llm, history(), {}, Ledger())

    assert result is not None and result.error is None and result.updates is not None
    assert {
        key: value["value"] for key, value in result.updates["working"].items.items()
    } == {"budget": "1.2", "platforms": "iOS, Android", "cash": "нет"}


@pytest.mark.asyncio
async def test_router_keeps_previous_keys_and_drops_nulled_ones():
    llm = StubLLM(
        '{"working": {"goal": {"value": "собрать ТЗ", "kind": "goal"}, "city": null}}'
    )
    previous = {
        "working": version({"name": item("Сергей", kind="step"), "city": item("Казань", kind="step")})
    }

    result = await router(("working",)).extract(llm, history(), previous, Ledger())

    assert result is not None and result.error is None and result.updates is not None
    update = result.updates["working"]
    assert sorted(update.items) == ["goal", "name"]
    # A removal is a change a human can be asked about, so it is offered too.
    assert update.changes["city"] == {"value": "", "kind": "step"}


@pytest.mark.asyncio
async def test_router_reports_budget_spent_on_reasoning():
    result = await router(("working",)).extract(StubLLM(""), history(), {}, Ledger())

    assert result is not None
    assert result.updates is None
    assert "пустую память" in (result.error or "")


def test_retired_facts_mode_becomes_a_window_plus_a_working_layer():
    spec = AgentSpec.model_validate(
        {
            "context": {
                "mode": "facts",
                "keep_last_messages": 4,
                "facts_model": "glm-5.3-flash",
                "facts_max_tokens": 500,
                "overflow": "fail",
            }
        }
    )

    assert spec.context.mode == "window"
    assert spec.context.keep_last_messages == 4
    assert spec.context.overflow == "fail"
    assert spec.memory.working_enabled is True
    assert spec.memory.model == "glm-5.3-flash"
    assert spec.memory.max_tokens == 500
    assert spec.memory.longterm_enabled is False


def test_each_layer_is_counted_on_its_own_and_the_sum_is_the_estimate():
    state = ContextState(
        role="роль",
        chars_per_token=3.0,
        summary={"version": 1, "text": "первый ход завершён", "tokens": 4},
        working=version({"task": item("собрать ТЗ")}),
        longterm=version({"name": item("Сергей", kind="profile")}),
    )

    view = SummaryContextStrategy(2, 2, "gpt-5-nano", 64).build(tail(), state)

    # Every block is measured as it is rendered, header included, so the
    # per-layer numbers add up to the estimate of the whole call.
    assert min(view.longterm_tokens, view.working_tokens, view.summary_tokens) > 0
    assert view.estimated_tokens == (
        view.role_tokens
        + view.longterm_tokens
        + view.working_tokens
        + view.summary_tokens
        + view.tail_tokens
    )
    rendered = view.render("chat")
    assert view.longterm_tokens == math.ceil(len(rendered[1]["content"]) / 3)
    assert view.working_tokens == math.ceil(len(rendered[2]["content"]) / 3)
    assert view.summary_tokens == math.ceil(len(rendered[3]["content"]) / 3)
    # The tail costs what the tail holds: memory no longer hides inside it.
    assert view.tail_tokens == math.ceil(sum(entry.chars() for entry in view.tail) / 3)


def test_breakdown_reaches_the_api_and_names_the_unattributed_rest():
    state = ContextState(
        role="роль",
        chars_per_token=3.0,
        working=version({"task": item("собрать ТЗ")}),
        longterm=version({"name": item("Сергей", kind="profile")}),
    )
    view = NoneContextStrategy().build(tail(), state)
    blocks = (
        view.role_tokens
        + view.longterm_tokens
        + view.working_tokens
        + view.tail_tokens
    )
    # The agent replaces the estimate with the measured request, which also
    # carries the transport envelope and the tool schemas.
    view.estimated_tokens = blocks + 17

    payload = view.as_dict(limit_tokens=1000, limit_source="spec", overflow="drop")

    assert payload["longterm"]["payload_tokens"] == view.longterm_tokens
    assert payload["working"]["payload_tokens"] == view.working_tokens
    assert payload["working"]["tokens"] == 4  # what the write cost, untouched
    assert payload["summary"] is None
    assert payload["tail"]["tokens"] == view.tail_tokens
    assert payload["overhead_tokens"] == 17


def test_an_emptied_layer_is_not_sent_as_an_empty_block():
    view = NoneContextStrategy().build(
        tail(),
        ContextState(
            role="роль",
            chars_per_token=3.0,
            working={"version": 2, "items": {}},
        ),
    )

    assert view.working_tokens == 0
    assert view.estimated_tokens == view.role_tokens + view.tail_tokens
    assert all(WORKING_HEADER not in item["content"] for item in view.render("responses"))
    # The version still exists and the panel still shows it — at zero cost.
    assert view.as_dict()["working"]["payload_tokens"] == 0


def test_layers_without_content_cost_nothing():
    view = NoneContextStrategy().build(
        tail(), ContextState(role="роль", chars_per_token=3.0)
    )

    assert (view.longterm_tokens, view.working_tokens, view.summary_tokens) == (0, 0, 0)
    assert view.estimated_tokens == view.role_tokens + view.tail_tokens
    assert view.as_dict()["overhead_tokens"] == 0


def test_layer_budget_cuts_the_oldest_keys_out_of_the_call_only():
    stored = version(
        {
            "цель": item("собрать ТЗ на мобильный клиент", seq=1),
            "срок": item("макет к среде", kind="constraint", seq=3),
            "блокер": item("нет доступа к дизайн-системе", kind="blocker", seq=5),
        }
    )
    view = NoneContextStrategy().build(
        tail(),
        ContextState(
            role="роль",
            chars_per_token=3.0,
            working=stored,
            working_max_tokens=40,
        ),
    )

    # The budget bounds the block, not the layer: what left the call is the
    # oldest key, and the stored version still has all three.
    assert view.working_tokens <= 40
    assert view.working_dropped == ["цель"]
    block = next(
        item["content"] for item in view.render("chat") if WORKING_HEADER in item["content"]
    )
    assert "собрать ТЗ" not in block and "макет к среде" in block
    assert view.as_dict()["working"]["dropped_keys"] == ["цель"]
    assert set(stored["items"]) == {"цель", "срок", "блокер"}


def test_layer_without_a_budget_is_sent_whole():
    stored = version({"цель": item("собрать ТЗ на мобильный клиент", seq=1)})

    view = NoneContextStrategy().build(
        tail(), ContextState(role="роль", chars_per_token=3.0, working=stored)
    )

    assert view.working_dropped == []
    assert view.working_tokens == math.ceil(
        len(f"{WORKING_HEADER}\n{view.working_body}") / 3
    )


def test_a_budget_smaller_than_one_key_drops_the_block_instead_of_billing_it():
    view = NoneContextStrategy().build(
        tail(),
        ContextState(
            role="роль",
            chars_per_token=3.0,
            working=version(
                {"цель": item("собрать полное ТЗ на мобильный клиент и согласовать", seq=1)}
            ),
            working_max_tokens=32,
        ),
    )

    assert view.working_tokens == 0
    assert view.working_dropped == ["цель"]
    assert all(WORKING_HEADER not in item["content"] for item in view.render("chat"))
