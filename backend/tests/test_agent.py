import pytest

from agentbox.core.agent import Agent
from agentbox.core.events import ContextPrepared, Delta, Done, Failed, UsageReported
from agentbox.core.memory import Conversation
from agentbox.core.spec import AgentSpec
from agentbox.llm.stub import StubLLM


async def collect(agent: Agent, text: str):
    return [event async for event in agent.stream(text)]


@pytest.mark.asyncio
async def test_streams_deltas_then_done():
    agent = Agent(AgentSpec(), StubLLM("Раз два три"))
    events = await collect(agent, "привет")

    assert [type(e) for e in events[:-1]] == [
        ContextPrepared,
        Delta,
        Delta,
        Delta,
        UsageReported,
    ]
    usage = events[-2]
    assert usage.estimate["request_tokens"] > 0
    assert usage.actual["input_tokens"] > 0
    assert usage.actual["output_tokens"] > 0
    assert usage.ledger["requests"] == 1
    done = events[-1]
    assert isinstance(done, Done)
    assert done.text.strip() == "Раз два три"
    assert done.ledger["requests"] == 1
    assert done.ledger["cost_usd"] > 0


@pytest.mark.asyncio
async def test_memory_carries_previous_turns():
    llm = StubLLM("ок")
    agent = Agent(AgentSpec(), llm)
    await collect(agent, "первый")
    await collect(agent, "второй")

    roles = [m["role"] for m in llm.requests[-1].messages]
    assert roles == ["system", "user", "assistant", "user"]


@pytest.mark.asyncio
async def test_trim_drops_oldest_but_keeps_last_question():
    agent = Agent(AgentSpec(memory_limit_tokens=100), StubLLM("ок"))
    for i in range(6):
        await collect(agent, f"вопрос {i} " + "х" * 300)

    assert agent._last_context is not None
    assert agent._last_context.estimated_tokens <= 100 * 1.5
    assert agent.conversation.messages[-2].role == "user"
    assert agent._last_context.dropped_seqs


@pytest.mark.asyncio
async def test_artificial_context_limit_trims_old_history_and_reports_it():
    agent = Agent(
        AgentSpec(memory_limit_tokens=10_000, context_limit_tokens=180),
        StubLLM("помню"),
    )
    agent.load_history(
        [
            {"role": "user", "content": "старый факт " + "х" * 240},
            {"role": "assistant", "content": "принято " + "у" * 240},
        ]
    )

    events = await collect(agent, "новый вопрос")

    usage = next(event for event in events if isinstance(event, UsageReported))
    assert usage.context["limit_tokens"] == 180
    assert usage.context["limit_source"] == "spec"
    assert usage.context["dropped_messages"] == 2
    assert len(agent.conversation.messages) == 4
    assert [entry.seq for entry in agent._last_context.tail] == [3]


@pytest.mark.asyncio
async def test_fail_overflow_does_not_call_provider_or_trim_history():
    llm = StubLLM("не вызван")
    agent = Agent(
        AgentSpec(
            memory_limit_tokens=10_000,
            context_limit_tokens=100,
            context_overflow="fail",
        ),
        llm,
    )
    agent.load_history([{"role": "user", "content": "старое " + "х" * 300}])

    events = await collect(agent, "новое " + "у" * 300)

    assert isinstance(events[-1], Failed)
    assert "Контекст не помещается" in events[-1].message
    assert llm.requests == []
    assert [entry.seq for entry in agent.conversation.messages] == [1]


@pytest.mark.asyncio
async def test_send_overflow_reaches_provider_without_truncation():
    llm = StubLLM(fail_with='HTTP 400 от Zen: {"error":"context_length"}')
    agent = Agent(
        AgentSpec(
            context={
                "mode": "tokens",
                "memory_limit_tokens": 10_000,
                "window_tokens": 100,
                "overflow": "send",
            },
        ),
        llm,
    )
    agent.load_history(
        [
            {"seq": 1, "role": "user", "content": "старое " + "х" * 300},
            {"seq": 2, "role": "assistant", "content": "ответ " + "у" * 300},
        ]
    )

    events = await collect(agent, "новое " + "я" * 300)

    assert isinstance(events[0], ContextPrepared)
    assert events[0].view["dropped_seqs"] == []
    assert isinstance(events[-1], Failed)
    assert "context_length" in events[-1].message
    assert [item["role"] for item in llm.requests[0].messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]


@pytest.mark.asyncio
async def test_failed_call_does_not_poison_memory():
    agent = Agent(AgentSpec(), StubLLM(fail_with="Zen недоступен"))
    events = await collect(agent, "привет")

    assert isinstance(events[-1], Failed)
    assert agent.conversation.messages == []


@pytest.mark.asyncio
async def test_budget_stops_further_calls():
    agent = Agent(AgentSpec(budget_usd=0.000_005), StubLLM("ок", cost=0.000_01))
    await collect(agent, "первый")
    events = await collect(agent, "второй")

    assert isinstance(events[-1], Failed)
    assert "бюджет" in events[-1].message


def test_calibration_moves_toward_measured_ratio():
    conversation = Conversation(role="роль", limit_tokens=1000)
    # The provider reports 100 tokens for 500 chars, so the true ratio is 5.0
    # while the estimate starts at 3.0; one call must move it partway there.
    conversation.calibrate(tokens_in=100, sent_chars=500)

    assert 3.0 < conversation.chars_per_token < 5.0


def _dialogue() -> Conversation:
    conversation = Conversation(role="роль", limit_tokens=4000)
    conversation.add("user", "прочитай a.txt")
    conversation.add_tool("call_1", "read_file", '{"path":"a.txt"}', "привет")
    conversation.add("assistant", "там «привет»")
    return conversation


def test_responses_render_keeps_tool_call_and_output_adjacent():
    items = _dialogue().responses_items()

    assert [item.get("type") or item["role"] for item in items] == [
        "system",
        "user",
        "function_call",
        "function_call_output",
        "assistant",
    ]
    assert items[2]["call_id"] == items[3]["call_id"] == "call_1"


def test_chat_render_states_the_same_dialogue_in_chat_spelling():
    items = _dialogue().chat_messages()

    assert [item["role"] for item in items] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    call = items[2]["tool_calls"][0]
    assert call["id"] == "call_1"
    assert call["function"] == {"name": "read_file", "arguments": '{"path":"a.txt"}'}
    assert items[3] == {"role": "tool", "tool_call_id": "call_1", "content": "привет"}


@pytest.mark.asyncio
async def test_agent_renders_history_for_the_transport_its_model_uses():
    """The model id decides the history format; nothing above memory branches."""
    chat = Agent(
        AgentSpec(model="glm-5.3-flash"), StubLLM("ок"), message_format="chat"
    )
    await collect(chat, "привет")
    assert chat._history() == chat.conversation.chat_messages()

    responses = Agent(AgentSpec(model="gpt-5-nano"), StubLLM("ок"))
    await collect(responses, "привет")
    assert responses._history() == responses.conversation.responses_items()


@pytest.mark.asyncio
async def test_summary_precedes_the_unsummarized_tail():
    llm = StubLLM("ок")
    agent = Agent(AgentSpec(context_strategy="summary"), llm)
    agent.load_summary("Пользователь выбрал синий код.")
    agent.load_history(
        [
            {"role": "user", "content": "продолжим"},
            {"role": "assistant", "content": "да"},
        ]
    )

    await collect(agent, "какой код?")

    sent = llm.requests[-1].messages
    assert [item["role"] for item in sent] == [
        "system",
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert "синий код" in sent[1]["content"]
