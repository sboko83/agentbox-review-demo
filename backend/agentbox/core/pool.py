"""In-process supervision for concurrent agent runs.

The pool does not turn an agent into a service.  It keeps ordinary ``Agent``
objects cheap, while placing the three shared resources outside them: provider
slots, a run-wide budget and subprocess executor slots.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import AsyncIterator, Callable

from agentbox.core.experiments import execute_cell
from agentbox.core.invariants import InvariantSet
from agentbox.core.ledger import BudgetExceeded, Ledger
from agentbox.core.ports import Chunk, LLMPort, LLMRequest, ToolCall, ToolExecutor, ToolResult, ToolSpec, Usage
from agentbox.core.spec import AgentSpec
from agentbox.core.user_profile import UserProfile
from agentbox.store import repository as repo


class SharedBudget:
    """One ledger for every model call in a run, guarded across coroutines."""

    def __init__(self, budget_usd: float | None) -> None:
        self.ledger = Ledger(budget_usd=budget_usd)
        self._lock = asyncio.Lock()
        self._exhausted = False

    async def before_call(self) -> None:
        async with self._lock:
            if self._exhausted:
                raise BudgetExceeded("общий бюджет прогона исчерпан")
            self.ledger.check_budget()

    async def record(self, usage: Usage) -> None:
        async with self._lock:
            self.ledger.record(usage)
            if (
                self.ledger.budget_usd is not None
                and self.ledger.cost_usd >= self.ledger.budget_usd
            ):
                self._exhausted = True

    async def exhausted(self) -> bool:
        async with self._lock:
            return self._exhausted

    async def snapshot(self) -> dict:
        async with self._lock:
            return {**self.ledger.snapshot(), "exhausted": self._exhausted}


class ProviderGate:
    """A process-wide semaphore around every provider stream."""

    def __init__(self, limit: int) -> None:
        if limit < 1:
            raise ValueError("Лимит параллельных вызовов провайдера должен быть положительным")
        self.limit = limit
        self._semaphore = asyncio.Semaphore(limit)
        self.active = 0
        self.waiting = 0

    def wrap(self, llm: LLMPort, budget: SharedBudget | None = None) -> "GatedLLM":
        return GatedLLM(llm, self, budget)

    def snapshot(self) -> dict:
        return {"limit": self.limit, "active": self.active, "waiting": self.waiting}


class GatedLLM:
    """LLM port decorator that shares the provider semaphore and run budget."""

    def __init__(self, llm: LLMPort, gate: ProviderGate, budget: SharedBudget | None) -> None:
        self._llm = llm
        self._gate = gate
        self._budget = budget

    async def aclose(self) -> None:
        """The application, not a decorator, owns the wrapped client's lifetime."""

    async def stream(self, request: LLMRequest) -> AsyncIterator[Chunk]:
        if self._budget is not None:
            await self._budget.before_call()
        self._gate.waiting += 1
        waiting = True
        try:
            async with self._gate._semaphore:
                self._gate.waiting -= 1
                waiting = False
                self._gate.active += 1
                try:
                    # A request could wait behind a call that exhausts the budget.
                    if self._budget is not None:
                        await self._budget.before_call()
                    usage: Usage | None = None
                    async for chunk in self._llm.stream(request):
                        if chunk.usage is not None:
                            usage = chunk.usage
                        yield chunk
                    if usage is not None and self._budget is not None:
                        await self._budget.record(usage)
                finally:
                    self._gate.active -= 1
        except BaseException:
            # Cancelled waiters never acquired the semaphore, so they still need
            # to leave the observable queue count.
            if waiting:
                self._gate.waiting -= 1
            raise


class ExecutorPool:
    """Shared queue for subprocess-backed tool calls from coding sessions."""

    def __init__(self, limit: int) -> None:
        if limit < 1:
            raise ValueError("Лимит executor'ов должен быть положительным")
        self.limit = limit
        self._semaphore = asyncio.Semaphore(limit)
        self.active = 0
        self.waiting = 0

    def wrap(self, executor: ToolExecutor) -> "QueuedExecutor":
        return QueuedExecutor(executor, self)

    def snapshot(self) -> dict:
        return {"limit": self.limit, "active": self.active, "waiting": self.waiting}


class QueuedExecutor:
    """Tool port decorator; only actual subprocess work occupies a slot."""

    def __init__(self, executor: ToolExecutor, pool: ExecutorPool) -> None:
        self._executor = executor
        self._pool = pool

    def describe(self) -> list[ToolSpec]:
        return self._executor.describe()

    def preview(self, call: ToolCall) -> str:
        return self._executor.preview(call)

    async def run(self, call: ToolCall) -> ToolResult:
        self._pool.waiting += 1
        waiting = True
        try:
            async with self._pool._semaphore:
                self._pool.waiting -= 1
                waiting = False
                self._pool.active += 1
                try:
                    return await self._executor.run(call)
                finally:
                    self._pool.active -= 1
        except BaseException:
            if waiting:
                self._pool.waiting -= 1
            raise


@dataclass(slots=True)
class PoolAgent:
    id: str
    config_index: int
    repeat_index: int
    name: str
    provider: str
    model: str
    state: str = "queued"
    error: str | None = None

    def snapshot(self) -> dict:
        return {
            "id": self.id,
            "config_index": self.config_index,
            "repeat_index": self.repeat_index,
            "name": self.name,
            "provider": self.provider,
            "model": self.model,
            "state": self.state,
            "error": self.error,
        }


class AgentRegistry:
    """In-memory ownership and state for live pool agents."""

    def __init__(self) -> None:
        self._agents: dict[str, dict[str, PoolAgent]] = {}

    def register(
        self,
        run_id: str,
        specs: list[AgentSpec],
        repeats: int,
        config_names: list[str] | None = None,
    ) -> list[PoolAgent]:
        agents = [
            PoolAgent(
                id=f"{run_id}:{config_index}:{repeat_index}",
                config_index=config_index,
                repeat_index=repeat_index,
                name=config_names[config_index] if config_names else f"Конфигурация {config_index + 1}",
                provider=spec.model.provider,
                model=spec.model.model,
            )
            for config_index, spec in enumerate(specs)
            for repeat_index in range(repeats)
        ]
        self._agents[run_id] = {agent.id: agent for agent in agents}
        return agents

    def get(self, run_id: str, agent_id: str) -> PoolAgent:
        return self._agents[run_id][agent_id]

    def snapshot(self, run_id: str) -> list[dict]:
        return [agent.snapshot() for agent in self._agents.get(run_id, {}).values()]


class RunSupervisor:
    """Starts confirmed matrices in the background and exposes their dashboard."""

    def __init__(self, conn, provider: ProviderGate, executors: ExecutorPool) -> None:
        self._conn = conn
        self.provider = provider
        self.executors = executors
        self.registry = AgentRegistry()
        self._budgets: dict[str, SharedBudget] = {}
        self._tasks: dict[str, asyncio.Task] = {}

    def start(
        self,
        *,
        run_id: str,
        provider_for: Callable[[AgentSpec], tuple[LLMPort, Callable[[str], str]]],
        prompt: str,
        specs: list[AgentSpec],
        repeats: int,
        budget_usd: float | None,
        config_names: list[str] | None = None,
        user_profiles: list[UserProfile | None] | None = None,
        invariant_sets: list[InvariantSet | None] | None = None,
    ) -> None:
        if run_id in self._tasks:
            raise ValueError("Прогон уже зарегистрирован")
        self.registry.register(run_id, specs, repeats, config_names)
        budget = SharedBudget(budget_usd)
        self._budgets[run_id] = budget
        self._tasks[run_id] = asyncio.create_task(
            self._run(
                run_id,
                provider_for,
                prompt,
                specs,
                repeats,
                budget,
                user_profiles,
                invariant_sets,
            ),
            name=f"agentbox-run-{run_id}",
        )

    async def _run(
        self,
        run_id: str,
        provider_for: Callable[[AgentSpec], tuple[LLMPort, Callable[[str], str]]],
        prompt: str,
        specs: list[AgentSpec],
        repeats: int,
        budget: SharedBudget,
        user_profiles: list[UserProfile | None] | None = None,
        invariant_sets: list[InvariantSet | None] | None = None,
    ) -> None:
        agents = self.registry.snapshot(run_id)
        spec_by_index = {index: spec for index, spec in enumerate(specs)}

        async def one(agent_data: dict) -> None:
            agent = self.registry.get(run_id, agent_data["id"])
            if await budget.exhausted():
                agent.state = "budget_stopped"
                repo.record_run_result(
                    self._conn,
                    run_id,
                    agent.config_index,
                    agent.repeat_index,
                    "budget_stopped",
                    _budget_stopped_result(),
                )
                return
            agent.state = "running"
            spec = spec_by_index[agent.config_index]
            try:
                llm, transport_for = provider_for(spec)
                result = await execute_cell(
                    self.provider.wrap(llm, budget),
                    self._conn,
                    run_id,
                    prompt,
                    spec,
                    transport_for=transport_for,
                    user_profile=(
                        user_profiles[agent.config_index] if user_profiles else None
                    ),
                    invariants=(
                        invariant_sets[agent.config_index] if invariant_sets else None
                    ),
                )
                repo.record_run_result(
                    self._conn,
                    run_id,
                    agent.config_index,
                    agent.repeat_index,
                    "budget_stopped"
                    if result["status"] == "error" and await budget.exhausted()
                    else result["status"],
                    result,
                )
                agent.state = (
                    "budget_stopped"
                    if result["status"] == "error" and await budget.exhausted()
                    else "completed"
                )
            except BudgetExceeded:
                agent.state = "budget_stopped"
                repo.record_run_result(
                    self._conn,
                    run_id,
                    agent.config_index,
                    agent.repeat_index,
                    "budget_stopped",
                    _budget_stopped_result(),
                )
            except Exception as error:
                # A lost cell would silently shrink the denominator of every
                # rate in the report, so a failure is persisted like any other
                # outcome instead of living only in the dashboard.
                agent.state = "failed"
                agent.error = str(error)
                repo.record_run_result(
                    self._conn,
                    run_id,
                    agent.config_index,
                    agent.repeat_index,
                    "failed",
                    _failed_result(str(error)),
                )

        try:
            await asyncio.gather(*(one(agent) for agent in agents))
        except Exception:
            repo.finish_run(self._conn, run_id, "failed")
            raise
        repo.finish_run(
            self._conn, run_id, "budget_exhausted" if await budget.exhausted() else "completed"
        )

    async def dashboard(self, run_id: str) -> dict:
        task = self._tasks.get(run_id)
        budget = self._budgets.get(run_id)
        return {
            "run_id": run_id,
            "active": bool(task and not task.done()),
            "agents": self.registry.snapshot(run_id),
            "budget": await budget.snapshot() if budget else None,
            "provider": self.provider.snapshot(),
            "executors": self.executors.snapshot(),
        }

    async def aclose(self) -> None:
        running = [(run_id, task) for run_id, task in self._tasks.items() if not task.done()]
        for _, task in running:
            task.cancel()
        if running:
            await asyncio.gather(*(task for _, task in running), return_exceptions=True)
            for run_id, _ in running:
                repo.finish_run(self._conn, run_id, "interrupted")


def _budget_stopped_result() -> dict:
    """Persist a transparent empty cell instead of silently dropping it."""
    return _empty_result("budget_stopped", "общий бюджет прогона исчерпан")


def _failed_result(error: str) -> dict:
    return _empty_result("failed", error)


def _empty_result(status: str, error: str) -> dict:
    return {
        "status": status,
        "answer": "",
        "judge": {
            "available": False,
            "passed": False,
            "error": error,
            "usage": {"tokens_in": 0, "tokens_out": 0, "reasoning": 0, "cost": 0},
        },
        "resolution": status,
        "attempts": [],
        "actual_cost_usd": 0,
    }
