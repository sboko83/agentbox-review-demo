"""The workspace boundary and the six tools."""

import pytest

from agentbox.core.ports import ToolCall
from agentbox.tools.local import LocalExecutor, ToolFailure


@pytest.fixture
def workspace(tmp_path):
    (tmp_path / "notes.md").write_text("alpha\nbeta\n", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "code.py").write_text("x = 1\n", encoding="utf-8")
    return tmp_path


@pytest.fixture
def executor(workspace):
    return LocalExecutor(workspace, timeout=10.0)


def call(tool: str, **arguments) -> ToolCall:
    import json

    return ToolCall(id=f"c-{tool}", tool=tool, arguments_json=json.dumps(arguments))


async def test_read_and_list(executor):
    read = await executor.run(call("read_file", path="notes.md"))
    assert read.ok and "alpha" in read.output

    listing = await executor.run(call("list_dir"))
    assert "sub/" in listing.output and "notes.md" in listing.output


async def test_grep_reports_file_and_line(executor):
    result = await executor.run(call("grep", pattern="bet.", glob="*.md"))
    assert result.ok
    assert result.output.startswith("notes.md:2:")


async def test_write_and_edit_change_the_file(executor, workspace):
    written = await executor.run(call("write_file", path="new.txt", content="hi"))
    assert written.ok
    assert (workspace / "new.txt").read_text(encoding="utf-8") == "hi"

    edited = await executor.run(
        call("edit_file", path="notes.md", old_text="beta", new_text="gamma")
    )
    assert edited.ok
    assert (workspace / "notes.md").read_text(encoding="utf-8") == "alpha\ngamma\n"


async def test_edit_refuses_an_ambiguous_fragment(executor, workspace):
    (workspace / "twice.txt").write_text("dup\ndup\n", encoding="utf-8")
    result = await executor.run(
        call("edit_file", path="twice.txt", old_text="dup", new_text="one")
    )
    assert not result.ok
    assert "встречается 2 раз" in result.output


@pytest.mark.parametrize("path", ["../outside.txt", "/etc/hosts", "sub/../../escape"])
async def test_paths_outside_the_workspace_are_refused(executor, path):
    result = await executor.run(call("read_file", path=path))
    assert not result.ok
    assert "выходит за пределы" in result.output


async def test_symlink_out_of_the_workspace_is_refused(executor, workspace, tmp_path):
    secret = tmp_path.parent / "secret.txt"
    secret.write_text("s3cret", encoding="utf-8")
    (workspace / "link.txt").symlink_to(secret)

    result = await executor.run(call("read_file", path="link.txt"))
    assert not result.ok
    assert "выходит за пределы" in result.output


async def test_command_runs_and_reports_exit_code(executor):
    result = await executor.run(call("run_command", command="wc -l notes.md"))
    assert result.ok and result.exit_code == 0
    assert "notes.md" in result.output


async def test_command_outside_the_allowlist_is_refused(executor):
    result = await executor.run(call("run_command", command="rm -rf /"))
    assert not result.ok
    assert "не в списке разрешённых" in result.output


async def test_command_timeout_kills_the_process(workspace):
    executor = LocalExecutor(workspace, timeout=0.2)
    result = await executor.run(
        call("run_command", command="python3 -c 'import time; time.sleep(5)'")
    )
    assert not result.ok
    assert "не уложилась" in result.output


async def test_broken_arguments_become_a_tool_error(executor):
    result = await executor.run(ToolCall(id="c1", tool="read_file", arguments_json="{"))
    assert not result.ok
    assert "не разбираются как JSON" in result.output


def test_preview_of_an_edit_is_a_diff(executor):
    preview = executor.preview(
        call("edit_file", path="notes.md", old_text="beta", new_text="gamma")
    )
    assert "-beta" in preview and "+gamma" in preview


def test_preview_of_a_command_is_the_command_line(executor):
    assert executor.preview(call("run_command", command="ls -a")) == "$ ls -a"


def test_missing_workspace_is_rejected_at_construction(tmp_path):
    with pytest.raises(ValueError):
        LocalExecutor(tmp_path / "nope")


def test_resolve_accepts_the_root_itself(executor, workspace):
    assert executor.resolve(".") == workspace.resolve()
    with pytest.raises(ToolFailure):
        executor.resolve("..")


async def test_allowlisted_but_missing_executable_is_a_tool_failure(workspace):
    """An absent binary is something the model can react to, not an OS error."""
    executor = LocalExecutor(workspace, timeout=10.0, allowed_commands=frozenset({"nosuchbin"}))

    result = await executor.run(call("run_command", command="nosuchbin --version"))

    assert not result.ok
    assert result.output == "исполняемый файл не найден: nosuchbin"
