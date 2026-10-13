"""Phase-5 pool limits are tested with a stub, never with a paid provider."""

import asyncio

import pytest

from agentbox.core.pool import ExecutorPool, ProviderGate, RunSupervisor
from agentbox.core.ports import ToolCall, ToolResult, ToolSpec
from agentbox.core.spec import AgentSpec
from agentbox.llm.stub import StubLLM
from agentbox.llm.zen import ZEN_CAPABILITIES
from agentbox.store import repository as repo
from agentbox.store.db import init_db


async def wait_for(supervisor: RunSupervisor, run_id: str) -> dict:
    while True:
        dashboard = await supervisor.dashboard(run_id)
        if not dashboard["active"]:
            return dashboard
        await asyncio.sleep(0)


def provider_for(llm):
    return lambda _spec: (llm, ZEN_CAPABILITIES.transport_for)


@pytest.mark.asyncio
async def test_pool_registers_every_agent_and_provider_semaphore_limits_streams(tmp_path):
    conn = init_db(tmp_path / "agentbox.db")
    specs = [AgentSpec(name=f"agent-{index}") for index in range(3)]
    run_id = repo.create_run(
        conn,
        name="parallel",
        prompt="задача",
        repeats=1,
        budget_usd=None,
        estimated_cost_usd=1,
        specs=specs,
    )
    llm = StubLLM("ответ", delay=0.001)
    supervisor = RunSupervisor(conn, ProviderGate(1), ExecutorPool(2))

    supervisor.start(
        run_id=run_id,
        provider_for=provider_for(llm),
        prompt="задача",
        specs=specs,
        repeats=1,
        budget_usd=None,
    )
    dashboard = await wait_for(supervisor, run_id)

    assert [agent["state"] for agent in dashboard["agents"]] == ["completed"] * 3
    assert llm.peak_active == 1
    assert repo.get_run(conn, run_id)["totals"]["requests"] == 6


@pytest.mark.asyncio
async def test_common_budget_stops_waiting_agents_before_another_provider_call(tmp_path):
    conn = init_db(tmp_path / "agentbox.db")
    specs = [AgentSpec(name=f"agent-{index}") for index in range(3)]
    run_id = repo.create_run(
        conn,
        name="budget",
        prompt="задача",
        repeats=1,
        budget_usd=0.00001,
        estimated_cost_usd=1,
        specs=specs,
    )
    llm = StubLLM("ответ", cost=0.00001)
    supervisor = RunSupervisor(conn, ProviderGate(1), ExecutorPool(2))

    supervisor.start(
        run_id=run_id,
        provider_for=provider_for(llm),
        prompt="задача",
        specs=specs,
        repeats=1,
        budget_usd=0.00001,
    )
    dashboard = await wait_for(supervisor, run_id)

    assert len(llm.requests) == 1
    assert [agent["state"] for agent in dashboard["agents"]] == [
        "completed",
        "budget_stopped",
        "budget_stopped",
    ]
    assert dashboard["budget"]["exhausted"] is True
    assert repo.get_run(conn, run_id)["status"] == "budget_exhausted"


@pytest.mark.asyncio
async def test_matrix_resolves_provider_for_each_configuration(tmp_path):
    conn = init_db(tmp_path / "agentbox.db")
    specs = [
        AgentSpec(model={"provider": "alpha", "model": "alpha-model"}),
        AgentSpec(model={"provider": "beta", "model": "beta-model"}),
    ]
    run_id = repo.create_run(
        conn,
        name="providers",
        prompt="задача",
        repeats=1,
        budget_usd=None,
        estimated_cost_usd=1,
        specs=specs,
    )
    clients = {"alpha": StubLLM("alpha"), "beta": StubLLM("beta")}
    supervisor = RunSupervisor(conn, ProviderGate(2), ExecutorPool(1))

    supervisor.start(
        run_id=run_id,
        provider_for=lambda spec: (
            clients[spec.model.provider],
            ZEN_CAPABILITIES.transport_for,
        ),
        prompt="задача",
        specs=specs,
        repeats=1,
        budget_usd=None,
    )
    await wait_for(supervisor, run_id)

    assert len(clients["alpha"].requests) == 2
    assert len(clients["beta"].requests) == 2
    assert clients["alpha"].requests[0].model == "alpha-model"
    assert clients["beta"].requests[0].model == "beta-model"


class SlowExecutor:
    def __init__(self) -> None:
        self.active = 0
        self.peak_active = 0

    def describe(self) -> list[ToolSpec]:
        return []

    def preview(self, call: ToolCall) -> str:
        return call.tool

    async def run(self, call: ToolCall) -> ToolResult:
        self.active += 1
        self.peak_active = max(self.peak_active, self.active)
        try:
            await asyncio.sleep(0.001)
            return ToolResult(call_id=call.id, output="ok")
        finally:
            self.active -= 1


@pytest.mark.asyncio
async def test_executor_pool_queues_tool_work():
    executor = SlowExecutor()
    queued = ExecutorPool(1).wrap(executor)

    await asyncio.gather(
        *(queued.run(ToolCall(id=str(index), tool="test")) for index in range(3))
    )

    assert executor.peak_active == 1


@pytest.mark.asyncio
async def test_a_crashed_cell_is_persisted_instead_of_vanishing(tmp_path):
    """A lost cell would silently shrink the denominator of every rate."""
    conn = init_db(tmp_path / "agentbox.db")
    specs = [AgentSpec(name="agent-0")]
    run_id = repo.create_run(
        conn,
        name="crash",
        prompt="задача",
        repeats=2,
        budget_usd=None,
        estimated_cost_usd=1,
        specs=specs,
    )

    class Exploding(StubLLM):
        async def stream(self, request):
            raise RuntimeError("провайдер лёг")
            yield  # pragma: no cover - makes this an async generator

    supervisor = RunSupervisor(conn, ProviderGate(2), ExecutorPool(1))
    supervisor.start(
        run_id=run_id,
        provider_for=provider_for(Exploding()),
        prompt="задача",
        specs=specs,
        repeats=2,
        budget_usd=None,
    )
    dashboard = await wait_for(supervisor, run_id)

    assert [agent["state"] for agent in dashboard["agents"]] == ["failed"] * 2
    report = repo.get_run(conn, run_id)
    assert [item["status"] for item in report["results"]] == ["failed"] * 2
    assert all("провайдер лёг" in item["judge"]["error"] for item in report["results"])
