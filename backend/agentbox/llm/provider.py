"""Provider composition and the single runtime registry.

Each provider owns three adapters: an optional configured client, a model
catalog and the capability map used to render requests.  The API resolves this
bundle by the provider name stored in AgentSpec; no provider-specific branch
exists outside ``llm/``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from importlib import import_module
from typing import Callable, Mapping, Protocol, cast

from agentbox.config import read_env_file
from agentbox.core.ports import LLMError, LLMPort

ProviderBuilder = Callable[[Mapping[str, str]], "Provider"]
PROVIDER_BUILDERS = {
    "zen": "agentbox.llm.zen:build_zen_provider",
}
# Price ceiling of the offered catalog, in dollars per million input tokens.
# Input dominates the bill of a long-context agent, so one axis is enough.
MAX_INPUT_USD_PER_MILLION = 5.0
MAX_PRICE_ENV = "AGENTBOX_MAX_MODEL_PRICE"


class ModelCatalog(Protocol):
    async def list(self) -> list[dict]: ...

    async def context_window(self, model_id: str) -> int | None: ...

    async def aclose(self) -> None: ...


class ProviderCapabilities(Protocol):
    def transport_for(self, model: str) -> str: ...

    def supported_params(self, model: str) -> frozenset[str]: ...


@dataclass(slots=True)
class Provider:
    name: str
    client: LLMPort | None
    catalog: ModelCatalog
    capabilities: ProviderCapabilities
    credential_env: str
    base_url: str

    @property
    def configured(self) -> bool:
        return self.client is not None

    async def aclose(self) -> None:
        if self.client is not None:
            await self.client.aclose()
        await self.catalog.aclose()


class ProviderRegistry:
    """Every configured provider plus the policy of what the catalog offers.

    The price ceiling belongs here, not in a vendor adapter: it is a choice
    about what this installation is willing to spend, and it must apply the
    same way to every provider added later.
    """

    def __init__(
        self,
        providers: Mapping[str, Provider],
        max_input_usd_per_million: float | None = MAX_INPUT_USD_PER_MILLION,
    ) -> None:
        self._providers = dict(providers)
        self._max_input_usd_per_million = max_input_usd_per_million

    def __getitem__(self, name: str) -> Provider:
        try:
            return self._providers[name]
        except KeyError as error:
            raise LLMError(f"Неизвестный LLM-провайдер: {name}") from error

    def require(self, name: str) -> Provider:
        provider = self[name]
        if provider.client is None:
            raise LLMError(
                f"Нет {provider.credential_env} для провайдера {name} — "
                "впишите ключ в secrets/.env и перезапустите."
            )
        return provider

    @property
    def status(self) -> dict[str, bool]:
        return {
            name: provider.configured for name, provider in self._providers.items()
        }

    async def list_models(self) -> list[dict]:
        configured = [
            provider for provider in self._providers.values() if provider.configured
        ]
        if not configured:
            variables = ", ".join(
                provider.credential_env for provider in self._providers.values()
            )
            raise LLMError(
                "Нет настроенных LLM-провайдеров — задайте один из ключей: "
                f"{variables}"
            )
        models: list[dict] = []
        errors: list[str] = []
        for provider in configured:
            try:
                items = await provider.catalog.list()
            except Exception as error:
                errors.append(f"{provider.name}: {error}")
                continue
            models.extend(
                {**item, "provider": provider.name, "offered": self._offered(item)}
                for item in items
            )
        if not models and errors:
            raise LLMError("; ".join(errors))
        return sorted(
            models,
            key=lambda item: (item["provider"], item["family"], item["id"]),
        )

    def _offered(self, item: Mapping[str, object]) -> bool:
        """A model too expensive to pick stays in the catalog but off the list.

        Removing the row instead would also remove its price and window, and
        the matrix estimate and the context meter of a session already using
        that model read them from here. An unknown price never hides a model:
        a models.dev outage must not empty the form.
        """
        if self._max_input_usd_per_million is None:
            return True
        price = item.get("input_usd_per_million")
        if not isinstance(price, (int, float)):
            return True
        return price < self._max_input_usd_per_million

    async def context_window(self, provider: str, model: str) -> int | None:
        return await self[provider].catalog.context_window(model)

    async def aclose(self) -> None:
        for provider in self._providers.values():
            await provider.aclose()


def build_provider_registry(values: Mapping[str, str] | None = None) -> ProviderRegistry:
    """Build every registered provider from environment plus secrets/.env."""
    merged = read_env_file()
    merged.update(os.environ)
    if values is not None:
        merged.update(values)
    return ProviderRegistry(
        {
            name: _load_builder(path)(merged)
            for name, path in PROVIDER_BUILDERS.items()
        },
        max_input_usd_per_million=_max_price(merged.get(MAX_PRICE_ENV)),
    )


def _max_price(raw: str | None) -> float | None:
    """Empty or unparsable ceiling means no ceiling, never a silent default."""
    if raw is None or not raw.strip():
        return MAX_INPUT_USD_PER_MILLION
    try:
        value = float(raw)
    except ValueError as error:
        raise ValueError(f"{MAX_PRICE_ENV} должен быть числом, получено {raw!r}") from error
    return value if value > 0 else None


def _load_builder(path: str) -> ProviderBuilder:
    module_name, separator, attribute = path.partition(":")
    if not separator:
        raise ValueError(f"Некорректный provider builder: {path}")
    return cast(ProviderBuilder, getattr(import_module(module_name), attribute))
