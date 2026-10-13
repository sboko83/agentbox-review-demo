"""Docker executor wiring without requiring a Docker daemon in the test suite."""

import asyncio
import json

import pytest

from agentbox.core.ports import ToolCall
from agentbox.tools.docker import DockerExecutor, DockerLimits


class ScriptExecutor(DockerExecutor):
    """Executes the generated container argv directly to validate tool scripts."""

    async def _container(self, argv: list[str]) -> tuple[str, int]:
        process = await asyncio.create_subprocess_exec(
            *argv,
            cwd=self.root,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        output, _ = await process.communicate()
        return output.decode().rstrip(), process.returncode


def call(tool: str, **arguments) -> ToolCall:
    return ToolCall(id=tool, tool=tool, arguments_json=json.dumps(arguments))


def test_docker_command_has_workspace_boundary_and_resource_limits(tmp_path):
    executor = DockerExecutor(
        tmp_path,
        image="agentbox-tools:test",
        limits=DockerLimits(cpus=1.5, memory_mb=384, pids=64),
    )

    argv = executor._docker_argv(["echo", "ok"])

    assert ["--network", "none"] == argv[argv.index("--network") : argv.index("--network") + 2]
    assert "--read-only" in argv
    assert argv[argv.index("--memory") + 1] == "384m"
    assert argv[argv.index("--pids-limit") + 1] == "64"
    assert (
        "type=bind,src=" + str(tmp_path.resolve()) + ",dst=/workspace"
        == argv[argv.index("--mount") + 1]
    )
    assert argv[-3:] == ["agentbox-tools:test", "echo", "ok"]


def test_docker_executor_keeps_the_same_tool_contract(tmp_path):
    limits = DockerLimits(cpus=0.1, memory_mb=64)
    executor = DockerExecutor(tmp_path, image="agentbox-tools:test", limits=limits)

    assert [tool.name for tool in executor.describe()] == [
        "read_file", "list_dir", "grep", "write_file", "edit_file", "run_command",
    ]


async def test_docker_executor_refuses_a_command_before_starting_a_container(tmp_path):
    executor = DockerExecutor(tmp_path, image="agentbox-tools:test", limits=DockerLimits())

    result = await executor.run(
        ToolCall(id="x", tool="run_command", arguments_json='{"command":"rm -rf /"}')
    )

    assert not result.ok
    assert "не в списке разрешённых" in result.output


async def test_docker_tool_scripts_preserve_workspace_tool_behavior(tmp_path):
    (tmp_path / "notes.md").write_text("alpha\nbeta\n", encoding="utf-8")
    executor = ScriptExecutor(tmp_path, image="agentbox-tools:test", limits=DockerLimits())

    assert "alpha" in (await executor.run(call("read_file", path="notes.md"))).output
    assert "notes.md" in (await executor.run(call("list_dir"))).output
    assert "notes.md:2:" in (await executor.run(call("grep", pattern="beta"))).output
    assert (await executor.run(call("write_file", path="new.txt", content="one"))).ok
    assert (await executor.run(
        call("edit_file", path="new.txt", old_text="one", new_text="two")
    )).ok
    assert (tmp_path / "new.txt").read_text(encoding="utf-8") == "two"


@pytest.mark.parametrize("kwargs", [{"cpus": 0}, {"memory_mb": 63}, {"pids": 15}])
def test_docker_limits_reject_an_unsafe_configuration(kwargs):
    with pytest.raises(ValueError):
        DockerLimits(**kwargs)


async def test_docker_command_output_is_formatted_like_the_local_executor(tmp_path):
    """Both executors implement one port, so one output shape, not two."""
    executor = ScriptExecutor(tmp_path, image="agentbox-tools:test", limits=DockerLimits())

    result = await executor.run(call("run_command", command="echo привет"))

    assert result.ok
    assert result.output.splitlines() == ["код выхода 0", "привет"]
    assert "\\n" not in result.output
