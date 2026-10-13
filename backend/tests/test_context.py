from agentbox.core.context import (
    ContextState,
    NoneContextStrategy,
    SummaryContextStrategy,
    TokensContextStrategy,
    WindowContextStrategy,
)
from agentbox.core.memory import Message, ToolExchange


def state(summary: dict | None = None) -> ContextState:
    return ContextState(role="роль", chars_per_token=3.0, summary=summary)


def dialogue() -> list[Message | ToolExchange]:
    return [
        Message("user", "вопрос 1", 1),
        ToolExchange("call-1", "read_file", "{}", "результат", 2),
        Message("assistant", "ответ 1", 3),
        Message("user", "вопрос 2", 4),
        Message("assistant", "ответ 2", 5),
        Message("user", "вопрос 3", 6),
    ]


def test_none_keeps_complete_history():
    view = NoneContextStrategy().build(dialogue(), state())

    assert [entry.seq for entry in view.tail] == [1, 2, 3, 4, 5, 6]
    assert view.dropped_seqs == []


def test_window_moves_boundary_to_question_and_keeps_tool_exchange_whole():
    view = WindowContextStrategy(keep_last_messages=4).build(dialogue(), state())

    assert [entry.seq for entry in view.tail] == [1, 2, 3, 4, 5, 6]
    responses = view.render("responses")
    kinds = [item.get("type") or item["role"] for item in responses]
    assert kinds[2:4] == ["function_call", "function_call_output"]


def test_window_drops_only_complete_question_answer_pairs():
    view = WindowContextStrategy(keep_last_messages=2).build(dialogue(), state())

    assert [entry.seq for entry in view.tail] == [4, 5, 6]
    assert view.dropped_seqs == [1, 2, 3]


def test_tokens_drops_complete_turns_until_view_fits():
    history = dialogue()
    history[0].content += "x" * 300
    view = TokensContextStrategy(memory_limit_tokens=6).build(history, state())

    assert [entry.seq for entry in view.tail] == [6]
    assert view.dropped_seqs == [1, 2, 3, 4, 5]


def test_summary_exposes_version_and_only_unsummarized_tail():
    summary = {
        "version": 2,
        "covered_seq": 3,
        "message_count": 3,
        "text": "Первый ход завершён.",
        "tokens": 8,
    }
    strategy = SummaryContextStrategy(3, 2, "gpt-5-nano", 64)

    view = strategy.build(dialogue(), state(summary))

    assert view.summary == summary
    assert [entry.seq for entry in view.tail] == [4, 5, 6]
    assert view.dropped_seqs == []
