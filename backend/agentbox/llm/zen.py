"""opencode Zen transport.

Knows the provider and the network: address, key, SSE stream, retries, where
the text lives in the payload and what the call cost. Knows nothing about
agents, dialogue or the interface.

Zen routes by vendor: `gpt-*` models are served by /responses, everything else
by /chat/completions. Both are streamed here; which one a model needs, and
which parameters it will accept, is decided by `ZenCapabilities` below.

Provider quirks, found the hard way in week-1 and week-2:
- a default urllib/httpx User-Agent gets cut by Cloudflare, so send our own;
- a provider error arrives with HTTP 200 and a body of {"type": "error", ...};
- on /responses usage comes in `response.completed`, while the price of the
  call arrives separately, in a final `ping` event with a `cost` field;
- on /chat/completions the price arrives *after* `data: [DONE]`, as one more
  data frame carrying `cost` — as a string. A parser that stops at `[DONE]`,
  the way the format's documentation suggests, loses the price of every call;
- function calls stream their arguments as deltas, and on /responses only the
  closing `response.output_item.done` carries a complete, parseable JSON
  string; on /chat/completions the fragments have to be glued by `index`;
- `delta.reasoning_content` on /chat/completions is the model thinking aloud,
  not its answer: emitting it would paste the reasoning into the dialogue.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, AsyncIterator, Mapping

import httpx

from agentbox.core.ports import Chunk, LLMError, LLMRequest, ToolCall, Usage

if TYPE_CHECKING:
    from agentbox.llm.provider import Provider

USER_AGENT = "agentbox (httpx)"
ZEN_BASE_URL = "https://opencode.ai/zen/v1"
ZEN_API_KEY_ENV = "OPENCODE_API_KEY"
ZEN_BASE_URL_ENV = "OPENCODE_BASE_URL"
RETRIES = 3
RETRY_PAUSE = 5.0
RETRYABLE_CODES = (429, 500, 502, 503, 504)

RESPONSES = "responses"
CHAT = "chat"
GATED_PARAMS = frozenset(
    {
        "temperature",
        "top_p",
        "top_k",
        "frequency_penalty",
        "presence_penalty",
        "stop",
        "seed",
        "reasoning_effort",
    }
)
_SUPPORTED: dict[str, frozenset[str]] = {
    RESPONSES: frozenset({"reasoning_effort"}),
    CHAT: GATED_PARAMS,
}

PATHS = {RESPONSES: "/responses", CHAT: "/chat/completions"}


class _Retryable(Exception):
    """Internal signal: the attempt may be repeated."""


class ZenCapabilities:
    """Zen model routing and the parameters accepted by each transport."""

    def transport_for(self, model: str) -> str:
        return RESPONSES if model.startswith("gpt-") else CHAT

    def supported_params(self, model: str) -> frozenset[str]:
        return _SUPPORTED[self.transport_for(model)]

    def accepts(self, model: str, param: str) -> bool:
        return param not in GATED_PARAMS or param in self.supported_params(model)


ZEN_CAPABILITIES = ZenCapabilities()


class ZenClient:
    """Async HTTP client for Zen — the only place that knows about the network."""

    def __init__(
        self,
        api_key: str,
        base_url: str = ZEN_BASE_URL,
        timeout: float = 180.0,
        client: httpx.AsyncClient | None = None,
        capabilities: ZenCapabilities = ZEN_CAPABILITIES,
    ) -> None:
        self._key = api_key
        self._base_url = base_url.rstrip("/")
        self._client = client or httpx.AsyncClient(timeout=timeout)
        self._owns_client = client is None
        self.capabilities = capabilities

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def stream(self, request: LLMRequest) -> AsyncIterator[Chunk]:
        """Send the conversation and yield answer chunks, usage last."""
        transport = self.capabilities.transport_for(request.model)
        payload = (
            _responses_payload(request, self.capabilities)
            if transport == RESPONSES
            else _chat_payload(request, self.capabilities)
        )
        parse = _parse_responses_sse if transport == RESPONSES else _parse_chat_sse

        last_error = "неизвестная ошибка"
        for attempt in range(RETRIES):
            if attempt:
                await asyncio.sleep(RETRY_PAUSE * attempt)
            # A retry is only safe while no text has been emitted: otherwise the
            # answer would be glued together from two different generations.
            emitted = False
            try:
                async for chunk in self._stream_once(PATHS[transport], payload, parse):
                    emitted = emitted or bool(chunk.text) or chunk.tool_call is not None
                    yield chunk
                return
            except _Retryable as error:
                if emitted:
                    raise LLMError(f"поток оборвался на середине: {error}") from error
                last_error = str(error)
        raise LLMError(last_error)

    async def _stream_once(self, path: str, payload: dict, parse) -> AsyncIterator[Chunk]:
        headers = {
            "User-Agent": USER_AGENT,
            "Authorization": f"Bearer {self._key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        try:
            async with self._client.stream(
                "POST", f"{self._base_url}{path}", json=payload, headers=headers
            ) as response:
                if response.status_code in RETRYABLE_CODES:
                    raise _Retryable(f"HTTP {response.status_code}")
                if response.status_code >= 400:
                    body = (await response.aread()).decode("utf-8", errors="replace")
                    raise LLMError(f"HTTP {response.status_code} от Zen: {_reason(body)}")
                async for chunk in parse(response.aiter_lines()):
                    yield chunk
        except httpx.HTTPError as error:
            raise _Retryable(f"сеть: {error}") from error


# --- request bodies -------------------------------------------------------


def _sampling(request: LLMRequest, capabilities: ZenCapabilities) -> dict:
    """Sampling knobs this model accepts, under their shared spelling."""
    offered = {
        "temperature": request.temperature,
        "top_p": request.top_p,
        "top_k": request.top_k,
        "frequency_penalty": request.frequency_penalty,
        "presence_penalty": request.presence_penalty,
        "seed": request.seed,
        "stop": list(request.stop) or None,
    }
    return {
        name: value
        for name, value in offered.items()
        if value is not None and capabilities.accepts(request.model, name)
    }


def _responses_payload(request: LLMRequest, capabilities: ZenCapabilities) -> dict:
    payload: dict = {
        "model": request.model,
        "input": request.messages,
        "stream": True,
        **_sampling(request, capabilities),
    }
    if request.max_output_tokens is not None:
        payload["max_output_tokens"] = request.max_output_tokens
    if request.text_format is not None:
        payload["text"] = {"format": request.text_format}
    if request.reasoning_effort and capabilities.accepts(
        request.model, "reasoning_effort"
    ):
        payload["reasoning"] = {"effort": request.reasoning_effort}
    if request.tools:
        payload["tools"] = [tool.as_function() for tool in request.tools]
    return payload


def _chat_payload(request: LLMRequest, capabilities: ZenCapabilities) -> dict:
    payload: dict = {
        "model": request.model,
        "messages": request.messages,
        "stream": True,
        **_sampling(request, capabilities),
    }
    if request.max_output_tokens is not None:
        payload["max_tokens"] = request.max_output_tokens
    if request.text_format is not None:
        payload["response_format"] = _chat_response_format(request.text_format)
    if request.reasoning_effort and capabilities.accepts(
        request.model, "reasoning_effort"
    ):
        payload["reasoning_effort"] = request.reasoning_effort
    if request.tools:
        payload["tools"] = [tool.as_chat_function() for tool in request.tools]
    return payload


def _chat_response_format(text_format: dict) -> dict:
    """Restate a /responses `text.format` as a chat `response_format`.

    The two endpoints carry the same JSON schema at different depths. Measured
    caveat: Zen accepts the chat spelling but does not enforce it — a chat
    model answered in prose to a strict schema, so callers must still cope with
    a non-JSON answer.
    """
    if text_format.get("type") != "json_schema":
        return text_format
    nested = {key: value for key, value in text_format.items() if key != "type"}
    nested.setdefault("strict", True)
    return {"type": "json_schema", "json_schema": nested}


def _reason(body: str) -> str:
    """Pull the human-readable message out of an error body."""
    try:
        return (json.loads(body).get("error") or {}).get("message") or body[:200]
    except (json.JSONDecodeError, AttributeError):
        return body[:200]


# --- stream parsers -------------------------------------------------------


async def _events(lines: AsyncIterator[str]) -> AsyncIterator[dict]:
    """Decode `data:` frames, ignoring `[DONE]` instead of stopping on it."""
    async for raw in lines:
        line = raw.strip()
        if not line.startswith("data:"):
            continue
        body = line[len("data:") :].strip()
        if not body or body == "[DONE]":
            continue
        try:
            yield json.loads(body)
        except json.JSONDecodeError:
            continue


async def _parse_responses_sse(lines: AsyncIterator[str]) -> AsyncIterator[Chunk]:
    """Turn the /responses SSE stream into text chunks plus a final usage chunk."""
    usage = Usage()
    async for event in _events(lines):
        kind = event.get("type")
        if kind == "response.output_text.delta":
            yield Chunk(text=event.get("delta", ""))
        elif kind == "response.output_item.done":
            # Arguments also arrive as deltas, but only the completed item is
            # guaranteed to hold valid JSON, so the loop waits for it.
            item = event.get("item") or {}
            if item.get("type") == "function_call":
                yield Chunk(
                    tool_call=ToolCall(
                        id=item.get("call_id") or item.get("id") or "",
                        tool=item.get("name") or "",
                        arguments_json=item.get("arguments") or "{}",
                    )
                )
        elif kind in {"response.completed", "response.incomplete"}:
            counts = (event.get("response") or {}).get("usage") or {}
            usage.tokens_in = counts.get("input_tokens") or 0
            usage.tokens_out = counts.get("output_tokens") or 0
            details = counts.get("output_tokens_details") or {}
            usage.reasoning = details.get("reasoning_tokens") or 0
        elif kind == "ping" and event.get("cost") is not None:
            usage.cost = float(event["cost"])
        elif kind == "error" or (kind or "").endswith(".failed"):
            message = (event.get("error") or {}).get("message") or "ошибка провайдера"
            raise _Retryable(message)
    yield Chunk(usage=usage)


async def _parse_chat_sse(lines: AsyncIterator[str]) -> AsyncIterator[Chunk]:
    """Turn the /chat/completions SSE stream into chunks plus a final usage chunk.

    `usage` repeats on nearly every frame as a running total, so the last one
    seen wins rather than being summed.
    """
    usage = Usage()
    calls: dict[int, ToolCall] = {}
    async for event in _events(lines):
        if event.get("type") == "error" or (event.get("error") and not event.get("choices")):
            message = (event.get("error") or {}).get("message") or "ошибка провайдера"
            raise _Retryable(message)

        counts = event.get("usage")
        if counts:
            usage.tokens_in = counts.get("prompt_tokens") or 0
            usage.tokens_out = counts.get("completion_tokens") or 0
            details = counts.get("completion_tokens_details") or {}
            usage.reasoning = details.get("reasoning_tokens") or 0
        if event.get("cost") is not None:
            usage.cost = float(event["cost"])

        for choice in event.get("choices") or []:
            delta = choice.get("delta") or {}
            if delta.get("content"):
                yield Chunk(text=delta["content"])
            for fragment in delta.get("tool_calls") or []:
                _merge_tool_call(calls, fragment)
            # The model stops asking for tools only once; emitting here keeps a
            # half-streamed call from reaching the agent.
            if choice.get("finish_reason") == "tool_calls":
                for call in _drain(calls):
                    yield Chunk(tool_call=call)

    for call in _drain(calls):
        yield Chunk(tool_call=call)
    yield Chunk(usage=usage)


def _merge_tool_call(calls: dict[int, ToolCall], fragment: dict) -> None:
    """Glue one `tool_calls` delta onto the call it belongs to.

    Only the first fragment of a call carries its id and name; the rest are
    argument text keyed by `index`, so the index is the identity here.
    """
    index = fragment.get("index") or 0
    call = calls.setdefault(index, ToolCall(id="", tool="", arguments_json=""))
    if fragment.get("id"):
        call.id = fragment["id"]
    function = fragment.get("function") or {}
    if function.get("name"):
        call.tool = function["name"]
    if function.get("arguments"):
        call.arguments_json += function["arguments"]


def _drain(calls: dict[int, ToolCall]) -> list[ToolCall]:
    """Hand over the collected calls and forget them."""
    collected = [calls[index] for index in sorted(calls)]
    calls.clear()
    for call in collected:
        call.arguments_json = call.arguments_json or "{}"
    return collected


def build_zen_provider(values: Mapping[str, str]) -> Provider:
    """Compose Zen from its own credential and overridable base URL."""
    from agentbox.llm.catalog import ZenModelCatalog
    from agentbox.llm.provider import Provider

    api_key = values.get(ZEN_API_KEY_ENV) or None
    base_url = values.get(ZEN_BASE_URL_ENV) or ZEN_BASE_URL
    client = (
        ZenClient(api_key, base_url=base_url, capabilities=ZEN_CAPABILITIES)
        if api_key
        else None
    )
    return Provider(
        name="zen",
        client=client,
        catalog=ZenModelCatalog(
            api_key,
            base_url=base_url,
            capabilities=ZEN_CAPABILITIES,
        ),
        capabilities=ZEN_CAPABILITIES,
        credential_env=ZEN_API_KEY_ENV,
        base_url=base_url,
    )
