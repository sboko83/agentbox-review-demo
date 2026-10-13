import pytest

from agentbox.core.ledger import Ledger
from agentbox.core.context import HELPER_REASONING_HEADROOM, SummaryContextStrategy, strategy_for
from agentbox.core.spec import AgentSpec
from agentbox.llm.stub import StubLLM


def history(count: int) -> list[dict]:
    return [
        {
            "seq": index + 1,
            "role": "user" if index % 2 == 0 else "assistant",
            "content": f"сообщение {index + 1}",
        }
        for index in range(count)
    ]


def summary_strategy(spec: AgentSpec) -> SummaryContextStrategy:
    strategy = strategy_for(spec.context)
    assert isinstance(strategy, SummaryContextStrategy)
    return strategy


@pytest.mark.asyncio
async def test_summary_waits_for_batch_then_keeps_recent_exchange():
    spec = AgentSpec(
        context_strategy="summary",
        keep_last_messages=3,
        summary_every=2,
        summary_model="gpt-5-nano",
        summary_max_tokens=64,
    )
    llm = StubLLM("Факт сохранён.")

    result = await summary_strategy(spec).summarize_if_due(
        llm, history(4), None, Ledger()
    )

    assert result is not None
    assert result.covered_seq == 2
    assert result.message_count == 2
    assert result.text == "Факт сохранён."
    assert llm.requests[0].model == "gpt-5-nano"
    assert llm.requests[0].max_output_tokens == 64 + HELPER_REASONING_HEADROOM
    assert llm.requests[0].tools == []


@pytest.mark.asyncio
async def test_summary_is_not_called_before_threshold():
    spec = AgentSpec(
        context_strategy="summary", keep_last_messages=3, summary_every=3
    )
    llm = StubLLM("не вызван")

    result = await summary_strategy(spec).summarize_if_due(
        llm, history(4), None, Ledger()
    )

    assert result is None
    assert llm.requests == []


@pytest.mark.asyncio
async def test_next_summary_includes_previous_version():
    spec = AgentSpec(
        context_strategy="summary", keep_last_messages=3, summary_every=2
    )
    llm = StubLLM("Новая сводка.")
    previous = {"covered_seq": 2, "message_count": 2, "text": "Старая сводка."}

    result = await summary_strategy(spec).summarize_if_due(
        llm, history(6), previous, Ledger()
    )

    assert result is not None
    assert result.covered_seq == 4
    assert result.message_count == 4
    assert "Старая сводка." in llm.requests[0].messages[1]["content"]
