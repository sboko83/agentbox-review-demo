"""An LLM port that answers without a network.

Used by tests, and later by phase 5 to check that a pool of agents scales:
what should limit the pool is the architecture, not the provider bill.
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator

from agentbox.core.ports import Chunk, LLMError, LLMRequest, ToolCall, Usage


class StubLLM:
    """Streams canned turns: text, and optionally tool calls before it.

    `script` is a list of turns; each turn is `(text, [ToolCall, ...])`. The
    stub walks it call by call, which is exactly what a tool-loop test needs:
    ask for a tool, then answer once its result is back.
    """

    def __init__(
        self,
        answer: str = "Готово.",
        *,
        script: list[tuple[str, list[ToolCall]]] | None = None,
        cost: float = 0.000_01,
        delay: float = 0.0,
        fail_with: str | None = None,
    ) -> None:
        self.script = script if script is not None else [(answer, [])]
        self.cost = cost
        self.delay = delay
        self.fail_with = fail_with
        self.requests: list[LLMRequest] = []
        self.active = 0
        self.peak_active = 0

    async def aclose(self) -> None:
        """Nothing to release; the port contract still requires the method."""

    async def stream(self, request: LLMRequest) -> AsyncIterator[Chunk]:
        turn = min(len(self.requests), len(self.script) - 1)
        self.requests.append(request)
        self.active += 1
        self.peak_active = max(self.peak_active, self.active)
        try:
            if self.fail_with:
                raise LLMError(self.fail_with)

            text, calls = self.script[turn]
            for word in text.split(" ") if text else []:
                if self.delay:
                    await asyncio.sleep(self.delay)
                yield Chunk(text=word + " ")
            # A model that got no tools cannot ask for one.
            for call in calls if request.tools else []:
                yield Chunk(tool_call=call)

            sent = sum(len(str(item.get("content") or item)) for item in request.messages)
            yield Chunk(
                usage=Usage(
                    tokens_in=max(1, sent // 3),
                    tokens_out=max(1, len(text) // 3),
                    cost=self.cost,
                )
            )
        finally:
            self.active -= 1
