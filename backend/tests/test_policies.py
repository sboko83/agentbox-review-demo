"""Phase-3 policy behaviour, independent from HTTP and a paid model."""

import json

import pytest

from agentbox.core.agent import Agent
from agentbox.core.policies import OutputPolicyContext
from agentbox.core.ports import ToolCall
from agentbox.core.spec import AgentSpec, migrate_spec_v1
from agentbox.llm.stub import StubLLM
from agentbox.tools.local import LocalExecutor


async def collect(agent: Agent, text: str):
    return [event async for event in agent.stream(text)]


@pytest.mark.parametrize(
    ("strategy", "calls", "first_needle", "last_needle"),
    [
        ("direct", 1, "задача", "задача"),
        ("step_by_step", 1, "Решай пошагово", "Решай пошагово"),
        ("self_prompt", 2, "самодостаточный промпт", "ответ 0"),
        ("experts", 4, "АНАЛИТИК", "ВЕДУЩИЙ группы экспертов"),
    ],
)
async def test_input_strategies_build_the_week_one_workflow(
    strategy, calls, first_needle, last_needle
):
    llm = StubLLM(
        script=[(f"ответ {index}", []) for index in range(calls)]
    )
    agent = Agent(AgentSpec(strategy=strategy), llm)

    events = await collect(agent, "задача")

    assert len(llm.requests) == calls
    assert first_needle in llm.requests[0].messages[-1]["content"]
    assert last_needle in llm.requests[-1].messages[-1]["content"]
    if strategy == "experts":
        assert "ОТВЕТ АНАЛИТИКА:\nответ 0" in llm.requests[2].messages[-1]["content"]
    assert events[-1].text.strip() == f"ответ {calls - 1}"


@pytest.mark.parametrize(
    ("control", "has_api_limit", "has_api_format", "prompt_needle"),
    [
        ("unrestricted", False, False, ""),
        ("api", True, True, ""),
        ("prompt", False, False, "Требования к ответу:"),
    ],
)
async def test_output_controls_change_the_provider_request(
    control, has_api_limit, has_api_format, prompt_needle
):
    llm = StubLLM("готово")
    agent = Agent(AgentSpec(response_control=control), llm)

    await collect(agent, "ответь")

    request = llm.requests[0]
    assert (request.max_output_tokens is not None) is has_api_limit
    assert (request.text_format is not None) is has_api_format
    assert prompt_needle in request.messages[-1]["content"]


class ObserveCalls:
    """A third-party output policy: records the exact tool request it sees."""

    def __init__(self):
        self.contexts: list[OutputPolicyContext] = []

    def apply(self, context: OutputPolicyContext) -> None:
        if context.phase == "after_llm":
            self.contexts.append(context)


async def test_output_policy_receives_requested_tool_calls(tmp_path):
    observer = ObserveCalls()
    llm = StubLLM(
        script=[("", [ToolCall(id="read", tool="read_file", arguments_json='{"path":"a.txt"}')]), ("ok", [])]
    )
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    agent = Agent(
        AgentSpec(approval_mode="auto"),
        llm,
        executor=LocalExecutor(tmp_path),
        output_policy=observer,
    )

    await collect(agent, "прочитай")

    assert [call.tool for call in observer.contexts[0].tool_calls] == ["read_file"]


def test_agent_spec_schema_exposes_every_phase_three_control():
    root = AgentSpec.model_json_schema()
    schema = root["properties"]

    assert set(
        ("model", "sampling", "output", "reasoning", "context", "memory", "tools", "task", "judge")
    ) == set(schema)
    output = root["$defs"]["OutputSection"]["properties"]
    reasoning = root["$defs"]["ReasoningSection"]["properties"]
    assert output["mode"]["enum"] == ["free", "api", "prompt"]
    assert output["format_json"]["when"] == {"field": "mode", "values": ["api"]}
    assert reasoning["strategy"]["enum"] == ["direct", "step_by_step", "self_prompt", "experts"]
    assert schema["context"]["discriminator"]["propertyName"] == "mode"


def test_schema_also_declares_the_phase_four_judge_contract():
    schema = AgentSpec.model_json_schema()["$defs"]["JudgeSection"]["properties"]

    assert schema["failure_action"]["enum"] == ["as_is", "retry", "fallback"]
    assert schema["threshold"]["minimum"] == 0
    with pytest.raises(ValueError, match="запасная модель"):
        AgentSpec(judge={"failure_action": "fallback"})


def test_response_format_is_canonical_in_the_immutable_spec_hash():
    compact = AgentSpec(output={"format_json": '{"type":"json_schema","name":"answer"}'})
    spaced = AgentSpec(output={"format_json": '{ "name": "answer", "type": "json_schema" }'})

    assert compact.output.format_json == spaced.output.format_json
    assert compact.hash == spaced.hash


def test_v1_migration_removes_legacy_names_and_preserves_controls():
    migrated = migrate_spec_v1(
        {
            "name": "legacy",
            "profile": "chat",
            "model": "glm-5.3-flash",
            "response_control": "prompt",
            "strategy": "step_by_step",
            "context_strategy": "trim",
            "context_overflow": "fail",
            "memory_limit_tokens": 1234,
            "workspace_path": "/tmp/workspace",
            "executor_kind": "docker",
        }
    )
    spec = AgentSpec.model_validate(migrated)

    assert set(spec.model_dump()) == {
        "model", "sampling", "output", "reasoning", "context", "memory", "tools",
        "task", "judge"
    }
    assert spec.model.model == "glm-5.3-flash"
    assert spec.output.mode == "prompt"
    assert spec.reasoning.strategy == "step_by_step"
    assert spec.context.mode == "tokens"
    assert spec.context.overflow == "fail"
    assert spec.context.memory_limit_tokens == 1234
    assert spec.tools.executor == "docker"


@pytest.mark.parametrize("control", ["api", "prompt"])
async def test_response_control_leaves_preparatory_stages_alone(control):
    """A self-written prompt squeezed into the answer's shape is not a prompt."""
    llm = StubLLM(script=[("промпт", []), ("итог", [])])
    agent = Agent(AgentSpec(strategy="self_prompt", response_control=control), llm)

    await collect(agent, "задача")

    stage, final = llm.requests
    assert stage.max_output_tokens is None and stage.text_format is None
    assert "Требования к ответу:" not in stage.messages[-1]["content"]
    if control == "api":
        assert final.max_output_tokens is not None and final.text_format is not None
    else:
        assert "Требования к ответу:" in final.messages[-1]["content"]
