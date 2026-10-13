"""Spend tracking: tokens, money and the limit that stops a run.

A matrix run over many configurations costs real money, and stopping it after
the fact is more expensive than refusing to start the next call.
"""

from __future__ import annotations

from dataclasses import dataclass

from agentbox.core.ports import Usage


class BudgetExceeded(RuntimeError):
    """The configured budget is spent; no further calls are allowed."""


@dataclass(slots=True)
class Ledger:
    """Running totals for one agent, session or run."""

    budget_usd: float | None = None
    requests: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    reasoning: int = 0
    cost_usd: float = 0.0

    def check_budget(self) -> None:
        if self.budget_usd is not None and self.cost_usd >= self.budget_usd:
            raise BudgetExceeded(
                f"бюджет исчерпан: потрачено ${self.cost_usd:.4f} "
                f"из ${self.budget_usd:.4f}"
            )

    def record(self, usage: Usage) -> None:
        self.requests += 1
        self.tokens_in += usage.tokens_in
        self.tokens_out += usage.tokens_out
        self.reasoning += usage.reasoning
        self.cost_usd += usage.cost

    def snapshot(self) -> dict:
        return {
            "requests": self.requests,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "reasoning": self.reasoning,
            "cost_usd": round(self.cost_usd, 6),
            "budget_usd": self.budget_usd,
        }
