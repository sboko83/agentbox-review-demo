"""Router candidates for the user profile: checked by schema, never automatic."""

import json

import pytest

from agentbox.core.context import (
    PROFILE_CONTEXT,
    MemoryRouter,
    _profile_changes,
)
from agentbox.core.ledger import Ledger
from agentbox.core.user_profile import UserProfile
from agentbox.llm.stub import StubLLM

from test_sessions_api import client, sse_events  # noqa: F401 - shared fixture

PROFILE = UserProfile(name="Сергей", language="ru", about="iOS-разработчик")
HISTORY = [
    {"seq": 1, "role": "user", "content": "Впредь отвечай по-английски."},
    {"seq": 2, "role": "assistant", "content": "Sure."},
]


def test_foreign_fields_and_invalid_values_are_refused():
    accepted, refused = _profile_changes(
        PROFILE,
        {
            "language": "en",
            "name": "Другой",
            "about": "Android-разработчик",
            "font": "mono",
            "tone": "rude",
            "max_words": 5,
            "structure": {"value": "bullets"},
            "verbosity": "normal",
        },
    )

    assert accepted == {
        "language": {"value": '"en"', "kind": "setting"},
        "structure": {"value": '"bullets"', "kind": "setting"},
    }
    # `verbosity` already is `normal`: an unchanged value is no candidate.
    assert sorted(refused) == [
        "profile.about",
        "profile.font",
        "profile.max_words",
        "profile.name",
        "profile.tone",
    ]


def test_list_value_is_normalized_by_the_schema():
    accepted, refused = _profile_changes(
        PROFILE, {"constraints": ["  без платных сервисов ", ""], "max_words": None}
    )

    assert refused == []
    assert json.loads(accepted["constraints"]["value"]) == ["без платных сервисов"]
    assert "max_words" not in accepted


@pytest.mark.asyncio
async def test_router_splits_profile_from_memory_layers():
    stub = StubLLM(
        '{"working": {}, "profile": {"language": "en", "color": "red"}, "other": {}}'
    )
    router = MemoryRouter("gpt-5-nano", 64, ("working", "profile"))

    result = await router.extract(
        stub, HISTORY, {"working": None}, Ledger(budget_usd=None), user_profile=PROFILE
    )

    assert result is not None and result.error is None
    assert result.updates["profile"].changes == {
        "language": {"value": '"en"', "kind": "setting"}
    }
    assert sorted(result.rejected) == ["other", "profile.color"]
    role, prompt = (item["content"] for item in stub.requests[0].messages)
    assert "Слой profile" in role and "language: ru|en|auto" in role
    assert "longterm с видом profile" in role
    assert '"language": "ru"' in prompt and PROFILE_CONTEXT not in prompt


@pytest.mark.asyncio
async def test_router_without_profile_target_reads_it_only():
    stub = StubLLM('{"working": {}}')
    router = MemoryRouter("gpt-5-nano", 64, ("working",))

    await router.extract(
        stub, HISTORY, {"working": None}, Ledger(budget_usd=None), user_profile=PROFILE
    )

    role, prompt = (item["content"] for item in stub.requests[0].messages)
    assert "Слой profile" not in role
    assert PROFILE_CONTEXT in prompt


def _session(client, **memory) -> str:
    profile = client.post(
        "/api/user-profiles", json={"name": "Сергей", "language": "ru", "is_default": True}
    ).json()
    session_id = client.post(
        "/api/sessions",
        json={
            "spec": {
                "context": {"mode": "window", "keep_last_messages": 2},
                "memory": {"max_tokens": 64, **memory},
            }
        },
    ).json()["id"]
    return session_id, profile


def test_profile_candidate_is_parked_even_under_auto_and_accept_makes_a_version(client):
    stub = StubLLM(
        script=[
            ("Sure.", []),
            ('{"longterm": {}, "profile": {"language": "en", "name": "X"}}', []),
            ("Hello.", []),
            ('{"longterm": {}}', []),
        ]
    )
    client.app.state.providers["zen"].client = stub
    session_id, profile = _session(client, longterm_enabled=True, longterm_write="auto")

    events = sse_events(
        client.post(
            f"/api/sessions/{session_id}/messages",
            json={"text": "Впредь отвечай по-английски."},
        )
    )

    parked = next(e for e in events if e["type"] == "memory_proposals")
    assert parked["layer"] == "profile"
    assert [(p["key"], p["value"]) for p in parked["proposals"]] == [("language", '"en"')]
    rejected = next(e for e in events if e["type"] == "memory_rejected")
    assert rejected["keys"] == ["profile.name"]
    detail = client.get(f"/api/sessions/{session_id}").json()
    assert detail["memory"]["profile"]["router"] is True
    candidate = detail["memory"]["profile"]["proposals"][0]
    # Nothing is written before a human says so.
    assert detail["user_profile"]["profile"]["language"] == "ru"

    accepted = client.post(
        f"/api/sessions/{session_id}/memory/proposals/{candidate['id']}",
        json={"decision": "accept"},
    ).json()

    assert accepted["user_profile"]["profile"]["language"] == "en"
    assert accepted["user_profile"]["version_hash"] != profile["version_hash"]
    assert accepted["memory"]["profile"]["proposals"] == []
    sse_events(client.post(f"/api/sessions/{session_id}/messages", json={"text": "Привет"}))
    system = "\n".join(
        item["content"] for item in stub.requests[2].messages if item["role"] == "system"
    )
    assert "Язык ответа: английский" in system
    usage = client.get(f"/api/sessions/{session_id}").json()["usage"]
    assert [row["user_profile_hash"] for row in usage if row["kind"] == "chat"] == [
        profile["version_hash"],
        accepted["user_profile"]["version_hash"],
    ]


def test_candidate_belongs_to_the_profile_not_to_any_session(client):
    client.app.state.providers["zen"].client = StubLLM(
        script=[("Ок.", []), ('{"working": {}, "profile": {"tone": "strict"}}', [])]
    )
    session_id, _ = _session(client, working_enabled=True)
    sse_events(client.post(f"/api/sessions/{session_id}/messages", json={"text": "Строже."}))
    candidate = client.get(f"/api/sessions/{session_id}").json()["memory"]["profile"][
        "proposals"
    ][0]
    stranger = client.post("/api/sessions", json={"user_profile_id": None}).json()["id"]

    refused = client.post(
        f"/api/sessions/{stranger}/memory/proposals/{candidate['id']}",
        json={"decision": "accept"},
    )
    rejected = client.post(
        f"/api/sessions/{session_id}/memory/proposals/{candidate['id']}",
        json={"decision": "reject"},
    ).json()

    assert refused.status_code == 404
    assert rejected["memory"]["profile"]["proposals"] == []
    assert rejected["user_profile"]["profile"]["tone"] == "neutral"


def test_without_router_there_is_no_learning(client):
    stub = StubLLM("Ок.")
    client.app.state.providers["zen"].client = stub
    session_id, _ = _session(client)

    sse_events(
        client.post(f"/api/sessions/{session_id}/messages", json={"text": "Пиши по-английски."})
    )

    assert len(stub.requests) == 1
    profile = client.get(f"/api/sessions/{session_id}").json()["memory"]["profile"]
    assert profile == {"enabled": True, "router": False, "proposals": []}
