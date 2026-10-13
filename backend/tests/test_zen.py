import json

import httpx
import pytest

from agentbox.core.ports import LLMRequest, ToolSpec
from agentbox.llm.zen import ZenClient

TOOL = ToolSpec(name="read_file", description="read", parameters={"type": "object"})


def sse(*frames: str) -> bytes:
    """Frames exactly as Zen sends them, including a bare `[DONE]`."""
    return "".join(f"data: {frame}\n\n" for frame in frames).encode()


def _run(body: bytes, request: LLMRequest):
    """Stream one canned SSE body and report the chunks, payload and path."""
    seen: dict = {}
    path: list[str] = []

    def handler(http_request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(http_request.content))
        path.append(http_request.url.path)
        return httpx.Response(200, content=body)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    async def collect():
        zen = ZenClient("key", client=client)
        try:
            return [chunk async for chunk in zen.stream(request)]
        finally:
            await client.aclose()

    return collect(), seen, path


async def test_responses_transport_sends_api_output_controls():
    stream, seen, path = _run(
        sse(
            '{"type":"response.output_text.delta","delta":"ok"}',
            '{"type":"response.completed","response":{"usage":{}}}',
        ),
        LLMRequest(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": "hi"}],
            max_output_tokens=123,
            text_format={"type": "json_schema", "name": "answer"},
        ),
    )
    chunks = await stream

    assert path == ["/zen/v1/responses"]
    assert [chunk.text for chunk in chunks if chunk.text] == ["ok"]
    assert seen["max_output_tokens"] == 123
    assert seen["text"] == {"format": {"type": "json_schema", "name": "answer"}}


async def test_responses_drops_parameters_the_model_refuses():
    """Measured: every gpt-* id on Zen answers 400 to temperature and top_p."""
    stream, seen, _ = _run(
        sse('{"type":"response.completed","response":{"usage":{}}}'),
        LLMRequest(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.5,
            top_p=0.9,
            top_k=10,
            seed=7,
            stop=("###",),
            reasoning_effort="low",
        ),
    )
    await stream

    assert set(seen) == {"model", "input", "stream", "reasoning"}
    assert seen["reasoning"] == {"effort": "low"}


async def test_responses_incomplete_event_still_reports_usage():
    stream, _, _ = _run(
        sse(
            '{"type":"response.incomplete","response":{"usage":{'
            '"input_tokens":25,"output_tokens":64,'
            '"output_tokens_details":{"reasoning_tokens":64}}}}',
            '{"type":"ping","cost":"0.00002685"}',
        ),
        LLMRequest(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": "hi"}],
            max_output_tokens=64,
        ),
    )

    chunks = await stream

    usage = chunks[-1].usage
    assert (usage.tokens_in, usage.tokens_out, usage.reasoning) == (25, 64, 64)
    assert usage.cost == pytest.approx(0.00002685)


async def test_chat_transport_streams_text_and_bills_after_done():
    """The price of a chat call arrives in a frame *after* `data: [DONE]`."""
    stream, seen, path = _run(
        sse(
            '{"choices":[{"index":0,"delta":{"reasoning_content":"вслух"}}]}',
            '{"choices":[{"index":0,"delta":{"content":"Четы"}}]}',
            '{"choices":[{"index":0,"delta":{"content":"ре."}}]}',
            '{"choices":[],"usage":{"prompt_tokens":25,"completion_tokens":74,'
            '"completion_tokens_details":{"reasoning_tokens":60}}}',
            "[DONE]",
            '{"choices":[],"cost":"0.00004075"}',
        ),
        LLMRequest(
            model="glm-5.3-flash",
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.5,
            top_k=20,
            stop=("###",),
            max_output_tokens=64,
            reasoning_effort="low",
            tools=[TOOL],
        ),
    )
    chunks = await stream

    assert path == ["/zen/v1/chat/completions"]
    # Reasoning is the model thinking aloud, not the answer: it must not leak
    # into the dialogue.
    assert "".join(chunk.text for chunk in chunks) == "Четыре."
    usage = chunks[-1].usage
    assert (usage.tokens_in, usage.tokens_out, usage.reasoning) == (25, 74, 60)
    assert usage.cost == pytest.approx(0.00004075)

    assert seen["max_tokens"] == 64
    assert seen["reasoning_effort"] == "low"
    assert (seen["temperature"], seen["top_k"], seen["stop"]) == (0.5, 20, ["###"])
    assert seen["tools"] == [TOOL.as_chat_function()]


async def test_chat_transport_glues_tool_call_fragments_by_index():
    stream, _, _ = _run(
        sse(
            '{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1",'
            '"function":{"name":"read_file","arguments":"{\\"pa"}}]}}]}',
            '{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,'
            '"function":{"arguments":"th\\":\\"a.txt\\"}"}}]}}]}',
            '{"choices":[{"index":0,"finish_reason":"tool_calls","delta":{}}]}',
            '{"choices":[],"usage":{"prompt_tokens":166,"completion_tokens":36}}',
            "[DONE]",
        ),
        LLMRequest(
            model="glm-5.3-flash",
            messages=[{"role": "user", "content": "hi"}],
            tools=[TOOL],
        ),
    )
    chunks = await stream

    calls = [chunk.tool_call for chunk in chunks if chunk.tool_call]
    assert len(calls) == 1
    assert calls[0].id == "call_1"
    assert calls[0].tool == "read_file"
    assert calls[0].arguments() == {"path": "a.txt"}


async def test_chat_transport_restates_json_schema_at_chat_depth():
    stream, seen, _ = _run(
        sse('{"choices":[],"usage":{}}', "[DONE]"),
        LLMRequest(
            model="glm-5.3-flash",
            messages=[{"role": "user", "content": "hi"}],
            text_format={"type": "json_schema", "name": "answer", "schema": {"type": "object"}},
        ),
    )
    await stream

    assert seen["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "answer", "schema": {"type": "object"}, "strict": True},
    }
