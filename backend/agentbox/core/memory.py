"""Dialogue memory: what the agent still remembers of the conversation.

The provider bills by token, so the window is measured in tokens, not
messages. Nothing here counts tokens exactly — that would need the model's
tokenizer — so the estimate is chars-per-token, recalibrated against the real
`input_tokens` the provider reports after every call.

A tool call and its output are one entry, not two. The provider rejects a
`function_call_output` whose `function_call` is missing, so trimming must never
be able to separate them.

The stored dialogue is vendor-neutral; only the two renderers at the bottom of
`Conversation` know that /responses wants a flat item list with `function_call`
entries while /chat/completions wants messages with `assistant.tool_calls` and
`role: tool`. This is the single place where the history format depends on the
vendor — everything above it stores one dialogue, not two.
"""

from __future__ import annotations

from dataclasses import dataclass, field

INITIAL_CHARS_PER_TOKEN = 3.0
CALIBRATION_WEIGHT = 0.3


@dataclass(slots=True)
class Message:
    role: str  # user | assistant
    content: str
    seq: int = 0

    def responses_items(self) -> list[dict]:
        return [{"role": self.role, "content": self.content}]

    # A plain message is spelled the same way by both transports.
    chat_items = responses_items

    def chars(self) -> int:
        return len(self.content)


@dataclass(slots=True)
class ToolExchange:
    """One tool call together with what it returned."""

    call_id: str
    tool: str
    arguments_json: str
    output: str
    seq: int = 0

    def responses_items(self) -> list[dict]:
        return [
            {
                "type": "function_call",
                "call_id": self.call_id,
                "name": self.tool,
                "arguments": self.arguments_json,
            },
            {
                "type": "function_call_output",
                "call_id": self.call_id,
                "output": self.output,
            },
        ]

    def chat_items(self) -> list[dict]:
        return [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": self.call_id,
                        "type": "function",
                        "function": {
                            "name": self.tool,
                            "arguments": self.arguments_json,
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": self.call_id, "content": self.output},
        ]

    def chars(self) -> int:
        return len(self.arguments_json) + len(self.output)


Entry = Message | ToolExchange


@dataclass(slots=True)
class Conversation:
    """The complete in-memory transcript plus its provider role."""

    role: str
    limit_tokens: int
    messages: list[Entry] = field(default_factory=list)
    chars_per_token: float = INITIAL_CHARS_PER_TOKEN

    def add(self, role: str, content: str, seq: int | None = None) -> None:
        self.messages.append(
            Message(role=role, content=content, seq=seq or self.next_seq())
        )

    def add_tool(
        self,
        call_id: str,
        tool: str,
        arguments_json: str,
        output: str,
        seq: int | None = None,
    ) -> None:
        self.messages.append(
            ToolExchange(
                call_id=call_id,
                tool=tool,
                arguments_json=arguments_json,
                output=output,
                seq=seq or self.next_seq(),
            )
        )

    def next_seq(self) -> int:
        return max((entry.seq for entry in self.messages), default=0) + 1

    def responses_items(self) -> list[dict]:
        """History for /responses: role first, then the remembered dialogue."""
        return self._render("responses_items")

    def chat_messages(self) -> list[dict]:
        """History for /chat/completions: same dialogue, chat spelling."""
        return self._render("chat_items")

    def _render(self, method: str) -> list[dict]:
        items: list[dict] = [{"role": "system", "content": self.role}]
        for entry in self.messages:
            items.extend(getattr(entry, method)())
        return items

    def sent_chars(self) -> int:
        return len(self.role) + sum(entry.chars() for entry in self.messages)

    def estimate_tokens(self) -> int:
        return int(self.sent_chars() / self.chars_per_token)

    def calibrate(self, tokens_in: int, sent_chars: int) -> None:
        """Move the chars-per-token estimate toward what the provider measured."""
        if tokens_in > 0 and sent_chars > 0:
            measured = sent_chars / tokens_in
            self.chars_per_token += CALIBRATION_WEIGHT * (
                measured - self.chars_per_token
            )
