"""The user profile block: where it sits in the payload and which calls get it."""

import math

from agentbox.core.context import (
    LONGTERM_HEADER,
    SUMMARY_HEADER,
    SUMMARY_ROLE,
    WORKING_HEADER,
    ContextState,
    SummaryContextStrategy,
)
from agentbox.core.memory import Message
from agentbox.core.user_profile import PROFILE_HEADER, UserProfile
from agentbox.llm.stub import StubLLM

from test_sessions_api import client, sse_events  # noqa: F401 - shared fixture

PROFILE = {"name": "Коротко", "language": "en", "verbosity": "brief"}


def _item(value: str, kind: str) -> dict:
    return {"value": value, "kind": kind, "origin": "router", "seq": 1}


def test_payload_order_is_role_profile_longterm_working_summary_tail():
    profile = UserProfile(**PROFILE)
    state = ContextState(
        role="роль",
        chars_per_token=3.0,
        summary={"version": 1, "text": "первый ход", "tokens": 2},
        working={"items": {"task": _item("собрать ТЗ", "goal")}},
        longterm={"items": {"job": _item("iOS-разработчик", "profile")}},
        user_profile=profile,
    )
    tail = [
        Message("user", "старый", 1),
        Message("assistant", "ответ", 2),
        Message("user", "новый", 3),
    ]

    view = SummaryContextStrategy(2, 2, "gpt-5-nano", 64).build(tail, state)
    rendered = view.render("chat")

    assert [item["content"].split("\n", 1)[0] for item in rendered[:5]] == [
        "роль",
        PROFILE_HEADER,
        LONGTERM_HEADER,
        WORKING_HEADER,
        SUMMARY_HEADER,
    ]
    assert [item["role"] for item in rendered[5:]] == ["user", "assistant", "user"]
    # The profile renders its own header: it is sent once, byte for byte.
    assert rendered[1]["content"] == profile.render()
    assert rendered[1]["content"].count(PROFILE_HEADER) == 1
    assert view.profile_tokens == math.ceil(len(profile.render()) / 3)
    assert view.estimated_tokens == (
        view.role_tokens
        + view.profile_tokens
        + view.longterm_tokens
        + view.working_tokens
        + view.summary_tokens
        + view.tail_tokens
    )
    payload = view.as_dict()["user_profile"]
    assert payload == {
        "name": "Коротко",
        "hash": profile.hash,
        "content": profile.render(),
        "payload_tokens": view.profile_tokens,
    }


def test_no_profile_sends_no_block():
    view = SummaryContextStrategy(2, 2, "gpt-5-nano", 64).build(
        [Message("user", "вопрос", 1)], ContextState(role="роль", chars_per_token=3.0)
    )

    assert [item["content"] for item in view.render("chat")] == ["роль", "вопрос"]
    assert view.profile_tokens == 0
    assert view.as_dict()["user_profile"] is None


def _profile(client, **fields) -> dict:
    return client.post(
        "/api/user-profiles", json={**PROFILE, **fields, "is_default": True}
    ).json()


def _system_text(request) -> str:
    return "\n".join(
        str(item.get("content")) for item in request.messages if item["role"] == "system"
    )


def test_preparatory_stage_and_final_answer_carry_the_profile(client):
    stub = StubLLM(script=[("промпт", []), ("ответ", [])])
    client.app.state.providers["zen"].client = stub
    created = _profile(client)
    session_id = client.post(
        "/api/sessions", json={"spec": {"reasoning": {"strategy": "self_prompt"}}}
    ).json()["id"]

    sse_events(client.post(f"/api/sessions/{session_id}/messages", json={"text": "вопрос"}))

    rendered = UserProfile(**PROFILE).render()
    assert len(stub.requests) == 2
    assert all(rendered in _system_text(request) for request in stub.requests)
    usage = client.get(f"/api/sessions/{session_id}").json()["usage"]
    assert [row["user_profile_hash"] for row in usage] == [created["version_hash"]]


def test_summary_goes_without_the_profile_and_cites_no_version(client):
    stub = StubLLM(script=[("один", []), ("два", []), ("сводка", []), ("три", [])])
    client.app.state.providers["zen"].client = stub
    created = _profile(client)
    session_id = client.post(
        "/api/sessions",
        json={
            "spec": {
                "context": {
                    "mode": "summary",
                    "keep_last_messages": 3,
                    "summary_every": 2,
                    "summary_max_tokens": 64,
                }
            }
        },
    ).json()["id"]
    for text in ("первый", "второй", "третий"):
        sse_events(client.post(f"/api/sessions/{session_id}/messages", json={"text": text}))

    summary = next(r for r in stub.requests if r.messages[0]["content"] == SUMMARY_ROLE)
    assert PROFILE_HEADER not in str(summary.messages)
    chats = [r for r in stub.requests if r is not summary]
    assert all(PROFILE_HEADER in _system_text(request) for request in chats)
    usage = client.get(f"/api/sessions/{session_id}").json()["usage"]
    assert [(row["kind"], row["user_profile_hash"]) for row in usage] == [
        ("chat", created["version_hash"]),
        ("chat", created["version_hash"]),
        ("summary", None),
        ("chat", created["version_hash"]),
    ]


def test_router_and_planner_read_the_profile(client):
    stub = StubLLM(script=[("план", []), ('{"working": {}}', [])])
    client.app.state.providers["zen"].client = stub
    created = _profile(client)
    session_id = client.post(
        "/api/sessions",
        json={
            "spec": {
                "context": {"mode": "window", "keep_last_messages": 2},
                "memory": {"working_enabled": True, "max_tokens": 64},
            }
        },
    ).json()["id"]
    client.post(f"/api/sessions/{session_id}/plan/start")

    sse_events(client.post(f"/api/sessions/{session_id}/messages", json={"text": "план?"}))

    planner, memory = stub.requests
    assert PROFILE_HEADER in _system_text(planner)
    # The router sees the profile as a layer it may propose changes to.
    assert "Слой profile, текущие значения" in memory.messages[1]["content"]
    assert '"language": "en"' in memory.messages[1]["content"]
    usage = client.get(f"/api/sessions/{session_id}").json()["usage"]
    assert [(row["kind"], row["user_profile_hash"]) for row in usage] == [
        ("chat", created["version_hash"]),
        ("memory", created["version_hash"]),
    ]


def test_ledger_cites_the_version_each_call_carried(client):
    stub = StubLLM("ответ")
    client.app.state.providers["zen"].client = stub
    created = _profile(client)
    session_id = client.post("/api/sessions", json={}).json()["id"]
    sse_events(client.post(f"/api/sessions/{session_id}/messages", json={"text": "раз"}))

    edited = client.put(
        f"/api/user-profiles/{created['id']}", json={**PROFILE, "language": "ru"}
    ).json()
    sse_events(client.post(f"/api/sessions/{session_id}/messages", json={"text": "два"}))
    client.patch(f"/api/sessions/{session_id}", json={"user_profile_id": None})
    sse_events(client.post(f"/api/sessions/{session_id}/messages", json={"text": "три"}))

    assert edited["version_hash"] != created["version_hash"]
    assert "Язык ответа: русский" in _system_text(stub.requests[1])
    assert PROFILE_HEADER not in _system_text(stub.requests[2])
    detail = client.get(f"/api/sessions/{session_id}").json()
    assert [row["user_profile_hash"] for row in detail["usage"]] == [
        created["version_hash"],
        edited["version_hash"],
        None,
    ]
    assert detail["context"]["user_profile"] is None
