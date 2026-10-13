"""The tool loop: what gets offered, what needs asking, what comes back."""

import json

import pytest

from agentbox.core.agent import READ_ONLY, REFUSED, Agent
from agentbox.core.events import ApprovalNeeded, Done, ToolFinished, ToolStarted
from agentbox.core.ports import ToolCall
from agentbox.core.spec import AgentSpec
from agentbox.llm.stub import StubLLM
from agentbox.tools.local import LocalExecutor


class Answers:
    """An approval port with a fixed opinion."""

    def __init__(self, verdict: bool) -> None:
        self.verdict = verdict
        self.asked: list[str] = []

    async def confirm(self, call: ToolCall, preview: str) -> bool:
        self.asked.append(preview)
        return self.verdict


@pytest.fixture
def workspace(tmp_path):
    (tmp_path / "notes.md").write_text("alpha\n", encoding="utf-8")
    return tmp_path


def call(tool: str, call_id: str = "c1", **arguments) -> ToolCall:
    return ToolCall(id=call_id, tool=tool, arguments_json=json.dumps(arguments))


async def collect(agent: Agent, text: str):
    return [event async for event in agent.stream(text)]


async def test_a_read_call_runs_and_its_output_reaches_the_next_request(workspace):
    llm = StubLLM(script=[("", [call("read_file", path="notes.md")]), ("Там alpha.", [])])
    agent = Agent(AgentSpec(), llm, executor=LocalExecutor(workspace))

    events = await collect(agent, "что в notes.md?")
    kinds = [type(e) for e in events]
    assert ToolStarted in kinds and ToolFinished in kinds
    assert ApprovalNeeded not in kinds  # read_file does not mutate anything

    assert len(llm.requests) == 2
    items = llm.requests[1].messages
    assert {"type": "function_call_output", "call_id": "c1", "output": "alpha\n"} in items
    assert events[-1].text.strip() == "Там alpha."


async def test_ask_mode_stops_before_a_write_and_a_yes_lets_it_through(workspace):
    llm = StubLLM(
        script=[("", [call("write_file", path="new.txt", content="hi")]), ("Записал.", [])]
    )
    approvals = Answers(True)
    agent = Agent(
        AgentSpec(approval_mode="ask"),
        llm,
        executor=LocalExecutor(workspace),
        approvals=approvals,
    )

    events = await collect(agent, "создай new.txt")
    asked = [e for e in events if isinstance(e, ApprovalNeeded)]
    assert len(asked) == 1 and asked[0].tool == "write_file"
    assert "+hi" in asked[0].preview  # the диалог shows a diff, not raw arguments
    assert (workspace / "new.txt").read_text(encoding="utf-8") == "hi"


async def test_a_refusal_reaches_the_model_and_leaves_the_file_alone(workspace):
    llm = StubLLM(
        script=[("", [call("write_file", path="new.txt", content="hi")]), ("Понял.", [])]
    )
    agent = Agent(
        AgentSpec(approval_mode="ask"),
        llm,
        executor=LocalExecutor(workspace),
        approvals=Answers(False),
    )

    events = await collect(agent, "создай new.txt")
    finished = next(e for e in events if isinstance(e, ToolFinished))
    assert not finished.ok and finished.output == REFUSED
    assert not (workspace / "new.txt").exists()


async def test_silence_counts_as_a_refusal(workspace):
    llm = StubLLM(script=[("", [call("write_file", path="x.txt", content="hi")]), ("ок", [])])
    agent = Agent(AgentSpec(approval_mode="ask"), llm, executor=LocalExecutor(workspace))

    events = await collect(agent, "пиши")
    assert next(e for e in events if isinstance(e, ToolFinished)).output == REFUSED


async def test_read_only_mode_hides_mutating_tools(workspace):
    llm = StubLLM(script=[("", [call("write_file", path="x.txt", content="hi")]), ("ок", [])])
    agent = Agent(
        AgentSpec(approval_mode="read-only"), llm, executor=LocalExecutor(workspace)
    )

    events = await collect(agent, "пиши")
    offered = {tool.name for tool in llm.requests[0].tools}
    assert offered == {"read_file", "list_dir", "grep"}
    assert next(e for e in events if isinstance(e, ToolFinished)).output == READ_ONLY


async def test_auto_mode_writes_without_asking(workspace):
    llm = StubLLM(script=[("", [call("write_file", path="x.txt", content="hi")]), ("ок", [])])
    agent = Agent(AgentSpec(approval_mode="auto"), llm, executor=LocalExecutor(workspace))

    events = await collect(agent, "пиши")
    assert not [e for e in events if isinstance(e, ApprovalNeeded)]
    assert (workspace / "x.txt").read_text(encoding="utf-8") == "hi"


async def test_step_limit_forces_a_final_answer_without_tools(workspace):
    # A model that never stops asking: the loop must, and its last call has no tools.
    llm = StubLLM(script=[("", [call("list_dir")]), ("", [call("list_dir")])])
    agent = Agent(
        AgentSpec(max_tool_steps=2, approval_mode="auto"),
        llm,
        executor=LocalExecutor(workspace),
    )

    events = await collect(agent, "смотри")
    assert isinstance(events[-1], Done)
    assert len(llm.requests) == 3
    assert llm.requests[-1].tools == []


async def test_without_a_workspace_no_tools_are_offered():
    llm = StubLLM("просто ответ")
    agent = Agent(AgentSpec(), llm)

    await collect(agent, "привет")
    assert llm.requests[0].tools == []


async def test_history_restores_a_tool_exchange(workspace):
    llm = StubLLM("ок")
    agent = Agent(AgentSpec(), llm, executor=LocalExecutor(workspace))
    agent.load_history(
        [
            {"role": "user", "content": "покажи файл"},
            {
                "role": "tool",
                "call_id": "c1",
                "tool": "read_file",
                "arguments_json": '{"path": "notes.md"}',
                "content": "alpha\n",
            },
            {"role": "assistant", "content": "там alpha"},
        ]
    )

    await collect(agent, "ещё раз")
    items = llm.requests[0].messages
    assert {"type": "function_call", "call_id": "c1", "name": "read_file",
            "arguments": '{"path": "notes.md"}'} in items
