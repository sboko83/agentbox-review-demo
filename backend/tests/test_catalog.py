import httpx

from agentbox.llm.catalog import ZenModelCatalog


def _catalog(ids: list[str]) -> ZenModelCatalog:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            assert request.headers["authorization"] == "Bearer key"
            return httpx.Response(200, json={"data": [{"id": item} for item in ids]})
        return httpx.Response(
            200,
            json={
                "opencode": {
                    "models": {
                        "gpt-5-nano": {
                            "name": "GPT 5 nano",
                            "cost": {"input": 0.05, "output": 0.4},
                            "limit": {"context": 400_000},
                        }
                    }
                }
            },
        )

    return ZenModelCatalog("key", client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))


async def test_catalog_joins_prices_and_context_window():
    catalog = _catalog(["gpt-5-nano"])
    try:
        assert await catalog.list() == [
            {
                "id": "gpt-5-nano",
                "name": "GPT 5 nano",
                "family": "gpt",
                "transport": "responses",
                "input_usd_per_million": 0.05,
                "output_usd_per_million": 0.4,
                "context_window": 400_000,
                "supported_params": ["reasoning_effort"],
            }
        ]
    finally:
        await catalog.aclose()


async def test_catalog_keeps_chat_models_and_groups_by_family():
    catalog = _catalog(["glm-5.3-flash", "gpt-5-nano", "deepseek-v4-pro"])
    try:
        models = await catalog.list()
    finally:
        await catalog.aclose()

    assert [model["id"] for model in models] == ["deepseek-v4-pro", "glm-5.3-flash", "gpt-5-nano"]
    assert [model["family"] for model in models] == ["deepseek", "glm", "gpt"]
    # A chat model accepts every sampling knob; an id models.dev never heard of
    # keeps a null window instead of vanishing from the list.
    chat = models[1]
    assert chat["transport"] == "chat"
    assert "top_k" in chat["supported_params"]
    assert chat["context_window"] is None


async def test_catalog_hides_models_this_key_cannot_call():
    catalog = _catalog(
        [
            "gpt-5-nano",
            "claude-opus-5",
            "gemini-3-flash",
            "grok-4.6",
            "muse-spark-1.3",
            "big-pickle",
            "deepseek-v4-flash-free",
        ]
    )
    try:
        assert [model["id"] for model in await catalog.list()] == ["gpt-5-nano"]
    finally:
        await catalog.aclose()
