from agentbox.core.ports import LLMError
import pytest

from agentbox.llm.provider import (
    MAX_INPUT_USD_PER_MILLION,
    Provider,
    ProviderRegistry,
    build_provider_registry,
)
from agentbox.llm.stub import StubLLM


class Catalog:
    def __init__(
        self,
        model: str,
        window: int = 1234,
        prices: list[float | None] | None = None,
    ) -> None:
        self.model = model
        self.window = window
        self.prices = [1] if prices is None else prices
        self.closed = False

    async def list(self) -> list[dict]:
        return [
            {
                "id": self.model if index == 0 else f"{self.model}-{index}",
                "name": self.model,
                "family": "test",
                "transport": "chat",
                "input_usd_per_million": price,
                "output_usd_per_million": 2,
                "context_window": self.window,
                "supported_params": [],
            }
            for index, price in enumerate(self.prices)
        ]

    async def context_window(self, model_id: str) -> int | None:
        return self.window if model_id == self.model else None

    async def aclose(self) -> None:
        self.closed = True


class Capabilities:
    def transport_for(self, model: str) -> str:
        return "chat"

    def supported_params(self, model: str) -> frozenset[str]:
        return frozenset()


def provider(
    name: str, *, configured: bool = True, prices: list[float | None] | None = None
) -> Provider:
    return Provider(
        name=name,
        client=StubLLM() if configured else None,
        catalog=Catalog(f"{name}-model", prices=prices),
        capabilities=Capabilities(),
        credential_env=f"{name.upper()}_API_KEY",
        base_url=f"https://{name}.example",
    )


async def test_registry_aggregates_configured_catalogs_with_provider_identity():
    registry = ProviderRegistry(
        {"alpha": provider("alpha"), "beta": provider("beta", configured=False)}
    )

    models = await registry.list_models()

    assert [(model["provider"], model["id"]) for model in models] == [
        ("alpha", "alpha-model")
    ]
    assert await registry.context_window("alpha", "alpha-model") == 1234
    assert registry.status == {"alpha": True, "beta": False}


async def test_registry_reports_unknown_and_unconfigured_providers():
    registry = ProviderRegistry({"alpha": provider("alpha", configured=False)})

    for name, expected in (("missing", "Неизвестный"), ("alpha", "ALPHA_API_KEY")):
        try:
            registry.require(name)
        except LLMError as error:
            assert expected in str(error)
        else:  # pragma: no cover - assertion branch
            raise AssertionError("provider lookup must fail")


async def test_zen_builder_reads_its_own_key_and_base_url():
    registry = build_provider_registry(
        {
            "OPENCODE_API_KEY": "test-key",
            "OPENCODE_BASE_URL": "https://zen.example/v1/",
        }
    )
    zen = registry.require("zen")
    try:
        assert zen.base_url == "https://zen.example/v1/"
        assert zen.client._base_url == "https://zen.example/v1"
        assert zen.catalog._base_url == "https://zen.example/v1"
    finally:
        await registry.aclose()


async def test_price_ceiling_marks_expensive_models_without_dropping_them():
    registry = ProviderRegistry(
        {"alpha": provider("alpha", prices=[1.0, 5.0, 30.0, None])},
        max_input_usd_per_million=MAX_INPUT_USD_PER_MILLION,
    )

    models = await registry.list_models()

    # The ceiling is inclusive, and a price models.dev does not know never
    # hides a callable model.
    assert [(model["id"], model["offered"]) for model in models] == [
        ("alpha-model", True),
        ("alpha-model-1", False),
        ("alpha-model-2", False),
        ("alpha-model-3", True),
    ]
    # Price and window of a capped model stay readable for the matrix estimate.
    assert models[2]["output_usd_per_million"] == 2
    assert models[2]["context_window"] == 1234


async def test_price_ceiling_can_be_switched_off():
    registry = ProviderRegistry(
        {"alpha": provider("alpha", prices=[30.0])}, max_input_usd_per_million=None
    )

    assert [model["offered"] for model in await registry.list_models()] == [True]


async def test_registry_reads_the_ceiling_from_the_environment():
    registry = build_provider_registry(
        {"OPENCODE_API_KEY": "test-key", "AGENTBOX_MAX_MODEL_PRICE": "0.5"}
    )
    try:
        assert registry._max_input_usd_per_million == 0.5
    finally:
        await registry.aclose()

    unlimited = build_provider_registry(
        {"OPENCODE_API_KEY": "test-key", "AGENTBOX_MAX_MODEL_PRICE": "0"}
    )
    try:
        assert unlimited._max_input_usd_per_million is None
    finally:
        await unlimited.aclose()

    with pytest.raises(ValueError, match="AGENTBOX_MAX_MODEL_PRICE"):
        build_provider_registry(
            {"OPENCODE_API_KEY": "test-key", "AGENTBOX_MAX_MODEL_PRICE": "дорого"}
        )
