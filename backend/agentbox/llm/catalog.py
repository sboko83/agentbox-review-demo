"""The model catalog shown by the settings form.

Zen is authoritative for what can be called. models.dev only enriches those
ids with public list prices and context-window sizes, so a price outage never
hides a callable model.
"""

from __future__ import annotations

import asyncio

import httpx

from agentbox.core.ports import LLMError
from agentbox.llm.provider import ProviderCapabilities
from agentbox.llm.zen import USER_AGENT, ZEN_BASE_URL, ZEN_CAPABILITIES

PRICING_URL = "https://models.dev/api.json"

# Families Zen lists but this key cannot call. Verified by calling each one:
# `claude-*`, `gemini-*`, `grok-*` and `muse-*` answer with an upstream 500/503
# "Internal server error"; `*-free` and `big-pickle` answer `MissingSessionID`
# ("free tier can only be used in OpenCode"). Offering a configuration that is
# guaranteed not to start is worse than not offering it at all.
UNAVAILABLE_PREFIXES = ("claude-", "gemini-", "grok-", "muse-", "big-")
UNAVAILABLE_SUFFIXES = ("-free",)


def is_callable(model_id: str) -> bool:
    return not model_id.startswith(UNAVAILABLE_PREFIXES) and not model_id.endswith(
        UNAVAILABLE_SUFFIXES
    )


def family_of(model_id: str) -> str:
    """The grouping key of the select: everything before the first version part."""
    return model_id.split("-", 1)[0]


class ZenModelCatalog:
    """Every Zen model both transports can actually drive.

    The remaining ids are shown as they are: probing seventy models with a
    trial request before every form render is too expensive, and a model that
    does fail is marked unavailable after its first failed attempt.
    """

    def __init__(
        self,
        api_key: str | None,
        base_url: str = ZEN_BASE_URL,
        client: httpx.AsyncClient | None = None,
        capabilities: ProviderCapabilities = ZEN_CAPABILITIES,
    ) -> None:
        self._key = api_key
        self._base_url = base_url.rstrip("/")
        self._client = client or httpx.AsyncClient(timeout=30.0)
        self._owns_client = client is None
        self._capabilities = capabilities
        self._context_windows: dict[str, int | None] = {}

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def list(self) -> list[dict]:
        if not self._key:
            raise LLMError("Нет OPENCODE_API_KEY — каталог моделей недоступен.")
        models_task = self._client.get(
            f"{self._base_url}/models",
            headers={"Authorization": f"Bearer {self._key}", "User-Agent": USER_AGENT},
        )
        pricing_task = self._client.get(PRICING_URL, headers={"User-Agent": USER_AGENT})
        models_response, pricing_response = await asyncio.gather(
            models_task, pricing_task, return_exceptions=True
        )
        if isinstance(models_response, Exception):
            raise LLMError(f"Не удалось получить модели Zen: {models_response}")
        if models_response.status_code >= 400:
            raise LLMError(f"Zen вернул HTTP {models_response.status_code} для /models")
        try:
            payload = models_response.json()
            ids = [
                item["id"]
                for item in payload.get("data", [])
                if is_callable(item.get("id", ""))
            ]
        except (TypeError, ValueError, KeyError) as error:
            raise LLMError("Zen вернул некорректный каталог моделей") from error

        prices: dict = {}
        if not isinstance(pricing_response, Exception) and pricing_response.status_code < 400:
            try:
                prices = pricing_response.json().get("opencode", {}).get("models", {})
            except (TypeError, ValueError):
                prices = {}

        result = []
        for model_id in sorted(ids, key=lambda item: (family_of(item), item)):
            meta = prices.get(model_id, {})
            cost = meta.get("cost") or {}
            result.append(
                {
                    "id": model_id,
                    "name": meta.get("name", model_id),
                    "family": family_of(model_id),
                    "transport": self._capabilities.transport_for(model_id),
                    "input_usd_per_million": cost.get("input"),
                    "output_usd_per_million": cost.get("output"),
                    # Phase 5 draws the fill bar against this; an id models.dev
                    # does not know keeps a null and gets no ceiling drawn.
                    "context_window": (meta.get("limit") or {}).get("context"),
                    "supported_params": sorted(
                        self._capabilities.supported_params(model_id)
                    ),
                }
            )
        self._context_windows = {
            model["id"]: model["context_window"] for model in result
        }
        return result

    async def context_window(self, model_id: str) -> int | None:
        """Return a cached model ceiling, loading the catalog on first use."""
        if model_id not in self._context_windows:
            await self.list()
        return self._context_windows.get(model_id)
