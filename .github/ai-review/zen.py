"""Minimal opencode Zen client: one non-streaming completion, stdlib only.

`gpt-*` models are served by `POST /responses`, the rest by `POST /chat/completions`.
Zen reports provider errors with HTTP 200 and `{"type": "error"}`, and Cloudflare
rejects urllib's default User-Agent, so both are handled here.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

BASE_URL = "https://opencode.ai/zen/v1/"
PRICES_URL = "https://models.dev/api.json"
USER_AGENT = "AIAC9-day32-review/1.0"
RATE_LIMIT_WAITS = (5, 15, 30)


class ZenError(RuntimeError):
    pass


def endpoint(model: str) -> str:
    return "responses" if model.startswith("gpt-") else "chat/completions"


def payload(model: str, messages: list[dict], max_tokens: int) -> dict:
    if endpoint(model) == "responses":
        return {"model": model, "input": messages, "max_output_tokens": max_tokens}
    return {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0}


def parse(model: str, raw: dict) -> dict:
    """Return {"text", "input_tokens", "output_tokens"} from either endpoint's body."""
    if raw.get("type") == "error" or raw.get("error"):
        error = raw.get("error") or raw
        raise ZenError(f"Zen error: {error.get('message', error) if isinstance(error, dict) else error}")
    usage = raw.get("usage") or {}
    if endpoint(model) == "responses":
        text = "".join(
            part.get("text", "")
            for item in raw.get("output", []) if item.get("type") == "message"
            for part in item.get("content", []) if part.get("type") == "output_text"
        )
        return {"text": text, "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0)}
    choices = raw.get("choices") or [{}]
    text = (choices[0].get("message") or {}).get("content") or ""
    return {"text": text, "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0)}


def complete(model: str, messages: list[dict], api_key: str, max_tokens: int = 4000,
             timeout: float = 300, opener=urllib.request.urlopen, sleep=time.sleep) -> dict:
    body = json.dumps(payload(model, messages, max_tokens)).encode()
    request = urllib.request.Request(
        BASE_URL + endpoint(model), data=body, method="POST",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json",
                 "User-Agent": USER_AGENT},
    )
    for wait in RATE_LIMIT_WAITS + (None,):
        try:
            with opener(request, timeout=timeout) as response:
                return parse(model, json.loads(response.read()))
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")[:300]
            if error.code != 429 or wait is None:
                raise ZenError(f"Zen HTTP {error.code}: {detail}") from error
            sleep(wait)
    raise AssertionError("unreachable")


def price(model: str, opener=urllib.request.urlopen) -> tuple[float, float] | None:
    """USD per 1M input/output tokens from models.dev; None when unavailable."""
    try:
        request = urllib.request.Request(PRICES_URL, headers={"User-Agent": USER_AGENT})
        with opener(request, timeout=20) as response:
            cost = json.loads(response.read())["opencode"]["models"][model]["cost"]
        return float(cost["input"]), float(cost["output"])
    except Exception:  # noqa: BLE001 - the price is a footnote, never a failure
        return None
