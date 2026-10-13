import asyncio
import json
import time

import pytest
from fastapi.testclient import TestClient

from agentbox.api.app import create_app
from agentbox.api.routes.sessions import HttpApprovals, TaskBoundExecutor
from agentbox.core.ports import ToolCall, ToolResult, Usage
from agentbox.core.spec import AgentSpec
from agentbox.llm.stub import StubLLM
from agentbox.llm.provider import Provider, ProviderRegistry
from agentbox.llm.zen import ZEN_CAPABILITIES
from agentbox.store import repository as repo


class Catalog:
    async def list(self):
        return [
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

    async def aclose(self):
        pass

    async def context_window(self, model_id):
        return 400_000 if model_id == "gpt-5-nano" else None


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTBOX_DB", str(tmp_path / "test.db"))
    app = create_app()
    with TestClient(app) as test_client:
        app.state.providers = ProviderRegistry(
            {
                "zen": Provider(
                    name="zen",
                    client=StubLLM("Ответ модели"),
                    catalog=Catalog(),
                    capabilities=ZEN_CAPABILITIES,
                    credential_env="OPENCODE_API_KEY",
                    base_url="https://example.test",
                )
            }
        )
        yield test_client


def sse_events(response) -> list[dict]:
    return [
        json.loads(line[len("data: ") :])
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]


def task_transition(client, session_id: str, stage: str, **extra):
    detail = client.get(f"/api/sessions/{session_id}").json()
    return client.post(
        f"/api/sessions/{session_id}/task/transition",
        json={
            "stage": stage,
            "expected_revision": detail["task"]["revision"],
            "expected_invariant_hash": detail["invariants"]["version_hash"],
            **extra,
        },
    )


def task_pause(client, session_id: str, paused: bool):
    task = client.get(f"/api/sessions/{session_id}").json()["task"]
    return client.post(
        f"/api/sessions/{session_id}/task/{'pause' if paused else 'resume'}",
        json={"expected_revision": task["revision"]},
    )


def task_step(client, session_id: str, index: int, **patch):
    task = client.get(f"/api/sessions/{session_id}").json()["task"]
    return client.patch(
        f"/api/sessions/{session_id}/task/steps/{index}",
        json={**patch, "expected_revision": task["revision"]},
    )


def test_chat_round_trip_persists_messages_and_spend(client):
    session_id = client.post("/api/sessions", json={"title": "тест"}).json()["id"]

    response = client.post(
        f"/api/sessions/{session_id}/messages", json={"text": "привет"}
    )
    assert response.status_code == 200
    events = sse_events(response)
    assert events[0]["type"] == "context"
    assert events[0]["view"]["tail"]["first_seq"] == 1
    # The measured request costs more than its blocks: envelope and schemas.
    assert events[0]["view"]["overhead_tokens"] > 0
    assert events[1]["type"] == "delta"
    assert events[-2]["type"] == "usage"
    assert events[-1]["type"] == "done"
    assert events[-1]["text"].strip() == "Ответ модели"

    stored = client.get(f"/api/sessions/{session_id}").json()
    assert [m["role"] for m in stored["messages"]] == ["user", "assistant"]
    assert stored["usage"][0]["message_id"] == stored["messages"][1]["id"]
    assert stored["usage"][0]["invariant_hash"] == stored["invariants"]["version_hash"]
    assert stored["totals"]["requests"] == 1
    assert stored["totals"]["cost_usd"] > 0
    assert stored["usage"][0]["request_estimate_tokens"] > 0
    assert stored["usage"][0]["input_tokens"] > 0
    assert stored["usage"][0]["context_limit_tokens"] == 400_000
    assert stored["usage"][0]["context_limit_source"] == "model"
    assert stored["usage"][0]["latency_ms"] >= 0
    assert stored["context"]["mode"] == "tokens"
    assert stored["context"]["tail"]["first_seq"] == 1
    row = client.app.state.db.execute(
        "SELECT l.spec_hash, l.message_id, s.spec_hash AS current_hash, m.role"
        " FROM ledger l JOIN sessions s ON s.id = l.session_id"
        " JOIN messages m ON m.id = l.message_id WHERE l.session_id = ?",
        (session_id,),
    ).fetchone()
    assert row["spec_hash"] == row["current_hash"]
    assert row["role"] == "assistant"


def test_invariant_checks_are_streamed_billed_and_persisted_before_answer(client):
    allow = json.dumps(
        {
            "decision": "allow",
            "rule_ids": [],
            "explanation": "конфликта нет",
            "alternative": None,
        },
        ensure_ascii=False,
    )
    client.app.state.providers["zen"].client = StubLLM(
        script=[(allow, []), ("Разрешённый ответ", []), (allow, [])]
    )
    session_id = client.post("/api/sessions", json={}).json()["id"]
    initial = client.get(f"/api/sessions/{session_id}").json()
    changed = client.put(
        f"/api/sessions/{session_id}/invariants",
        json={
            "expected_version_hash": initial["invariants"]["version_hash"],
            "rules": [
                {
                    "id": "stack.python",
                    "category": "stack",
                    "text": "Использовать Python 3.9+.",
                }
            ],
        },
    ).json()

    response = client.post(
        f"/api/sessions/{session_id}/messages", json={"text": "Напиши пример"}
    )

    events = sse_events(response)
    kinds = [event["type"] for event in events]
    assert kinds[:3] == ["invariant_check", "context", "invariant_check"]
    assert kinds.index("delta") > kinds.index("invariant_check", 1)
    detail = client.get(f"/api/sessions/{session_id}").json()
    assert [item["target"] for item in detail["invariant_checks"]] == [
        "request",
        "response",
    ]
    assert all(item["outcome"] == "allow" for item in detail["invariant_checks"])
    assert [item["kind"] for item in detail["usage"]] == [
        "invariant",
        "invariant",
        "chat",
    ]
    assert all(
        item["invariant_hash"] == changed["invariants"]["version_hash"]
        for item in detail["usage"]
    )
    assert "[stack.python]" in detail["context"]["invariants"]["content"]


def test_context_calibration_survives_agent_recreation(client):
    session_id = client.post("/api/sessions", json={}).json()["id"]
    client.post(f"/api/sessions/{session_id}/messages", json={"text": "кириллица" * 20})
    calibrated = client.app.state.db.execute(
        "SELECT chars_per_token FROM sessions WHERE id = ?", (session_id,)
    ).fetchone()[0]

    client.post(f"/api/sessions/{session_id}/messages", json={"text": "второй"})

    assert calibrated != 3.0
    assert client.get(f"/api/sessions/{session_id}").json()["usage"][1][
        "input_estimate_tokens"
    ] > 0


def test_fail_overflow_is_reported_before_provider_call(client):
    created = client.post(
        "/api/sessions",
        json={
            "spec": {
                "context_limit_tokens": 100,
                "context_overflow": "fail",
                "memory_limit_tokens": 10_000,
            }
        },
    )
    session_id = created.json()["id"]

    response = client.post(
        f"/api/sessions/{session_id}/messages", json={"text": "х" * 500}
    )

    events = sse_events(response)
    assert events[-1]["type"] == "error"
    assert "Контекст не помещается" in events[-1]["message"]
    assert client.app.state.providers["zen"].client.requests == []
    assert client.get(f"/api/sessions/{session_id}").json()["messages"] == []


def test_history_survives_a_new_agent_instance(client):
    session_id = client.post("/api/sessions", json={}).json()["id"]
    client.post(f"/api/sessions/{session_id}/messages", json={"text": "первый"})
    client.post(f"/api/sessions/{session_id}/messages", json={"text": "второй"})

    # Each request rebuilds the agent from storage, so the second call must
    # still carry the first exchange.
    sent = client.app.state.providers["zen"].client.requests[-1].messages
    assert [m["role"] for m in sent] == ["system", "user", "assistant", "user"]


def test_session_context_view_uses_strategy_without_deleting_history(client):
    session_id = client.post(
        "/api/sessions",
        json={"spec": {"context": {"mode": "window", "keep_last_messages": 2}}},
    ).json()["id"]
    client.post(f"/api/sessions/{session_id}/messages", json={"text": "первый"})
    client.post(f"/api/sessions/{session_id}/messages", json={"text": "второй"})

    detail = client.get(f"/api/sessions/{session_id}").json()

    assert [message["seq"] for message in detail["messages"]] == [1, 2, 3, 4]
    assert [message["seq"] for message in detail["context"]["tail"]["messages"]] == [
        3,
        4,
    ]
    assert detail["context"]["dropped_seqs"] == [1, 2]


def test_session_context_view_restores_drop_overflow_boundary(client):
    session_id = client.post(
        "/api/sessions",
        json={
            "spec": {
                "model": {"role": "r"},
                "context": {
                    "mode": "none",
                    "window_tokens": 100,
                    "overflow": "drop",
                },
            }
        },
    ).json()["id"]
    for text in ("a" * 160, "b" * 160, "c" * 160):
        client.post(f"/api/sessions/{session_id}/messages", json={"text": text})

    detail = client.get(f"/api/sessions/{session_id}").json()

    assert detail["context"]["dropped_seqs"]
    assert detail["context"]["tail"]["first_seq"] > 1
    assert detail["context"]["estimated_tokens"] <= 100


def test_summary_is_versioned_billed_and_restored_before_recent_messages(client):
    client.app.state.providers["zen"].client = StubLLM(
        script=[
            ("первый ответ", []),
            ("второй ответ", []),
            ("Секретный код — СИНИЙ.", []),
            ("Код СИНИЙ.", []),
            ("Секретный код — СИНИЙ; диалог продолжен.", []),
            ("Продолжаем.", []),
        ]
    )
    session_id = client.post(
        "/api/sessions",
        json={
            "spec": {
                "context_strategy": "summary",
                "keep_last_messages": 3,
                "summary_every": 2,
                "summary_max_tokens": 64,
            }
        },
    ).json()["id"]
    client.post(f"/api/sessions/{session_id}/messages", json={"text": "код СИНИЙ"})
    client.post(f"/api/sessions/{session_id}/messages", json={"text": "продолжим"})

    response = client.post(
        f"/api/sessions/{session_id}/messages", json={"text": "какой код?"}
    )

    events = sse_events(response)
    summary_event = next(event for event in events if event["type"] == "summary")
    assert summary_event["version"] == 1
    assert summary_event["covered_seq"] == 2
    client.post(f"/api/sessions/{session_id}/messages", json={"text": "ещё шаг"})
    detail = client.get(f"/api/sessions/{session_id}").json()
    assert [item["version"] for item in detail["summaries"]] == [1, 2]
    assert [item["message_count"] for item in detail["summaries"]] == [2, 4]
    assert detail["summaries"][0]["summary_tokens"] > 0
    assert [row["kind"] for row in detail["usage"]] == [
        "chat",
        "chat",
        "summary",
        "chat",
        "summary",
        "chat",
    ]
    final_request = client.app.state.providers["zen"].client.requests[-1]
    assert [item["role"] for item in final_request.messages] == [
        "system",
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert "Секретный код" in final_request.messages[1]["content"]


def test_working_memory_is_versioned_billed_streamed_and_sent_before_recent_messages(client):
    client.app.state.providers["zen"].client = StubLLM(
        script=[
            ("Запомнил.", []),
            ('{"working":{"name":{"value":"Сергей","kind":"artifact"}}}', []),
            ("Привет, Сергей.", []),
            ('{"working":{"language":{"value":"Kotlin","kind":"constraint"}}}', []),
        ]
    )
    session_id = client.post(
        "/api/sessions",
        json={
            "spec": {
                "context": {"mode": "window", "keep_last_messages": 2},
                "memory": {"working_enabled": True, "max_tokens": 64},
            }
        },
    ).json()["id"]

    first = sse_events(
        client.post(
            f"/api/sessions/{session_id}/messages",
            json={"text": "Меня зовут Сергей."},
        )
    )

    memory_event = next(event for event in first if event["type"] == "memory")
    assert memory_event["layer"] == "working"
    assert memory_event["version"] == 1
    assert memory_event["covered_seq"] == 2
    assert memory_event["origin"] == "router"
    assert memory_event["items"]["name"]["value"] == "Сергей"
    assert first[-1]["type"] == "done"

    client.post(
        f"/api/sessions/{session_id}/messages", json={"text": "Я выбрал Kotlin."}
    )
    detail = client.get(f"/api/sessions/{session_id}").json()

    assert [item["version"] for item in detail["memory"]["working"]["versions"]] == [1, 2]
    assert detail["memory"]["longterm"] == {
        "enabled": False,
        "scope": "default",
        "sessions": 0,
        "write": "confirm",
        "versions": [],
        "proposals": [],
    }
    assert {
        key: item["value"]
        for key, item in detail["context"]["working"]["items"].items()
    } == {"name": "Сергей", "language": "Kotlin"}
    assert detail["context"]["longterm"] is None
    # The layer costs the next call what it costs, separately from the tail.
    context = detail["context"]
    assert context["working"]["payload_tokens"] > 0
    assert context["estimated_tokens"] == (
        context["role"]["tokens"]
        + context["working"]["payload_tokens"]
        + context["tail"]["tokens"]
        + context["overhead_tokens"]
    )
    assert [row["kind"] for row in detail["usage"]] == [
        "chat", "memory", "chat", "memory"
    ]
    second_chat_request = client.app.state.providers["zen"].client.requests[2]
    assert [item["role"] for item in second_chat_request.messages] == [
        "system", "system", "user", "assistant", "user"
    ]
    assert "Рабочая память" in second_chat_request.messages[1]["content"]
    second_fact_request = client.app.state.providers["zen"].client.requests[3]
    assert "Меня зовут" not in second_fact_request.messages[1]["content"]
    assert "Я выбрал Kotlin" in second_fact_request.messages[1]["content"]


def test_switching_working_memory_on_extracts_full_history_once(client):
    session_id = client.post("/api/sessions", json={}).json()["id"]
    client.post(
        f"/api/sessions/{session_id}/messages", json={"text": "Меня зовут Сергей."}
    )
    before = client.get(f"/api/sessions/{session_id}").json()
    before["spec"]["context"] = {
        "mode": "window",
        "keep_last_messages": 6,
        "window_tokens": None,
        "overflow": "drop",
        "budget_usd": None,
    }
    before["spec"]["memory"] = {**before["spec"]["memory"], "working_enabled": True, "max_tokens": 64}
    client.app.state.providers["zen"].client = StubLLM(
        '{"working":{"name":{"value":"Сергей","kind":"artifact"}}}'
    )

    switched = client.patch(
        f"/api/sessions/{session_id}/spec", json=before["spec"]
    )

    assert switched.status_code == 200
    detail = switched.json()
    assert detail["context"]["mode"] == "window"
    working = detail["memory"]["working"]["versions"][0]
    assert working["items"]["name"]["value"] == "Сергей"
    assert working["origin"] == "router"
    assert [row["kind"] for row in detail["usage"]] == ["chat", "memory"]
    assert "Меня зовут Сергей" in client.app.state.providers["zen"].client.requests[0].messages[1]["content"]

    client.app.state.providers["zen"].client = StubLLM("не должен вызываться")
    detail["spec"]["context"]["keep_last_messages"] = 4
    repeated = client.patch(
        f"/api/sessions/{session_id}/spec", json=detail["spec"]
    )
    assert repeated.status_code == 200
    assert client.app.state.providers["zen"].client.requests == []


def test_manual_memory_edit_keeps_the_kind_of_every_item(client):
    session_id = client.post(
        "/api/sessions", json={"spec": {"memory": {"working_enabled": True}}}
    ).json()["id"]

    edited = client.put(
        f"/api/sessions/{session_id}/memory/working",
        json={
            "items": {
                "name": {"value": "Сергей", "kind": "artifact"},
                "deadline": {"value": "среда", "kind": "constraint"},
            }
        },
    )
    cleared = client.put(
        f"/api/sessions/{session_id}/memory/working", json={"items": {}}
    )

    assert edited.status_code == 200
    first = edited.json()["memory"]["working"]["versions"][0]
    assert first["version"] == 1 and first["origin"] == "manual"
    # One layer, several kinds: rewriting it does not relabel what it holds.
    assert first["items"] == {
        "name": {"value": "Сергей", "kind": "artifact", "origin": "manual", "seq": 0},
        "deadline": {
            "value": "среда", "kind": "constraint", "origin": "manual", "seq": 0
        },
    }
    assert cleared.status_code == 200
    versions = cleared.json()["memory"]["working"]["versions"]
    assert [item["version"] for item in versions] == [1, 2]
    assert cleared.json()["context"]["working"]["items"] == {}
    assert cleared.json()["usage"] == []


def test_manual_rewrite_refuses_a_foreign_kind_and_keeps_an_untouched_route(client):
    client.app.state.providers["zen"].client = StubLLM(
        script=[
            ("Понял.", []),
            ('{"working":{"deadline":{"value":"среда","kind":"constraint"}}}', []),
        ]
    )
    session_id = client.post(
        "/api/sessions", json=_memory_spec(working_enabled=True)
    ).json()["id"]
    sse_events(
        client.post(f"/api/sessions/{session_id}/messages", json={"text": "к среде"})
    )

    foreign = client.put(
        f"/api/sessions/{session_id}/memory/working",
        json={"items": {"deadline": {"value": "среда", "kind": "profile"}}},
    )
    rewritten = client.put(
        f"/api/sessions/{session_id}/memory/working",
        json={
            "items": {
                "deadline": {"value": "среда", "kind": "constraint"},
                "goal": {"value": "макет", "kind": "goal"},
            }
        },
    )

    assert foreign.status_code == 422
    items = rewritten.json()["context"]["working"]["items"]
    # The untouched item still says the router put it there; only the new one
    # is hand-written.
    assert items["deadline"]["origin"] == "router"
    assert items["goal"] == {
        "value": "макет", "kind": "goal", "origin": "manual", "seq": 2
    }


def test_remembering_one_message_points_the_item_at_its_seq(client):
    client.app.state.providers["zen"].client = StubLLM(script=[("Ответ.", [])])
    session_id = client.post(
        "/api/sessions", json={"spec": {"memory": {"working_enabled": True}}}
    ).json()["id"]
    sse_events(
        client.post(f"/api/sessions/{session_id}/messages", json={"text": "цель — макет"})
    )

    written = client.post(
        f"/api/sessions/{session_id}/memory/working/items",
        json={"key": "goal", "value": "макет к среде", "kind": "goal", "seq": 1},
    )
    forgotten = client.delete(f"/api/sessions/{session_id}/memory/working/items/goal")
    missing = client.delete(f"/api/sessions/{session_id}/memory/working/items/goal")

    assert written.status_code == 200
    stored = written.json()["context"]["working"]["items"]["goal"]
    assert stored == {
        "value": "макет к среде", "kind": "goal", "origin": "manual", "seq": 1
    }
    assert forgotten.status_code == 200
    assert forgotten.json()["context"]["working"]["items"] == {}
    assert missing.status_code == 404
    # Forgetting is a new version, not an erased one.
    assert [
        item["version"] for item in forgotten.json()["memory"]["working"]["versions"]
    ] == [1, 2]


def test_moving_an_item_writes_both_layers_and_keeps_its_source_seq(client):
    session_id = client.post(
        "/api/sessions",
        json=_memory_spec(
            working_enabled=True, longterm_enabled=True, longterm_write="manual"
        ),
    ).json()["id"]
    client.post(
        f"/api/sessions/{session_id}/memory/working/items",
        json={"key": "stack", "value": "Kotlin Multiplatform", "kind": "artifact"},
    )

    moved = client.post(
        f"/api/sessions/{session_id}/memory/working/items/stack/move",
        json={"kind": "decision"},
    )
    foreign = client.post(
        f"/api/sessions/{session_id}/memory/longterm/items/stack/move",
        json={"kind": "decision"},
    )

    assert moved.status_code == 200
    detail = moved.json()
    assert detail["context"]["working"]["items"] == {}
    assert detail["context"]["longterm"]["items"]["stack"] == {
        "value": "Kotlin Multiplatform", "kind": "decision", "origin": "promote", "seq": 0
    }
    assert detail["memory"]["working"]["versions"][-1]["origin"] == "promote"
    assert detail["memory"]["longterm"]["versions"][-1]["origin"] == "promote"
    # The kind has to belong to the layer the item lands in.
    assert foreign.status_code == 422


def test_moving_refuses_a_layer_the_spec_keeps_off(client):
    session_id = client.post(
        "/api/sessions", json={"spec": {"memory": {"working_enabled": True}}}
    ).json()["id"]
    client.post(
        f"/api/sessions/{session_id}/memory/working/items",
        json={"key": "stack", "value": "Kotlin", "kind": "artifact"},
    )

    refused = client.post(
        f"/api/sessions/{session_id}/memory/working/items/stack/move",
        json={"kind": "decision"},
    )

    assert refused.status_code == 409
    assert "longterm" in refused.json()["detail"]


def test_finishing_a_task_promotes_the_marked_and_empties_the_working_layer(client):
    session_id = client.post(
        "/api/sessions",
        json=_memory_spec(
            working_enabled=True, longterm_enabled=True, longterm_write="manual"
        ),
    ).json()["id"]
    for key, value, kind in (
        ("stack", "Kotlin Multiplatform", "artifact"),
        ("deadline", "среда", "constraint"),
    ):
        client.post(
            f"/api/sessions/{session_id}/memory/working/items",
            json={"key": key, "value": value, "kind": kind},
        )

    finished = client.post(
        f"/api/sessions/{session_id}/memory/working/finish",
        json={"promote": {"stack": "decision"}},
    )

    assert finished.status_code == 200
    detail = finished.json()
    # The task took its own data with it; what was marked outlived it.
    assert detail["context"]["working"]["items"] == {}
    assert detail["context"]["longterm"]["items"] == {
        "stack": {
            "value": "Kotlin Multiplatform",
            "kind": "decision",
            "origin": "promote",
            "seq": 0,
        }
    }
    working = detail["memory"]["working"]["versions"]
    assert [item["version"] for item in working] == [1, 2, 3]
    # Emptying is a version, not a deletion: the finished task stays readable.
    assert working[-1]["items"] == {} and working[-1]["origin"] == "promote"
    assert working[-2]["items"]["deadline"]["value"] == "среда"


def test_finishing_refuses_an_unknown_key_and_a_long_term_layer_that_is_off(client):
    session_id = client.post(
        "/api/sessions", json={"spec": {"memory": {"working_enabled": True}}}
    ).json()["id"]
    client.post(
        f"/api/sessions/{session_id}/memory/working/items",
        json={"key": "stack", "value": "Kotlin", "kind": "artifact"},
    )

    missing = client.post(
        f"/api/sessions/{session_id}/memory/working/finish",
        json={"promote": {"goal": "decision"}},
    )
    foreign = client.post(
        f"/api/sessions/{session_id}/memory/working/finish",
        json={"promote": {"stack": "goal"}},
    )
    off = client.post(
        f"/api/sessions/{session_id}/memory/working/finish",
        json={"promote": {"stack": "decision"}},
    )
    emptied = client.post(
        f"/api/sessions/{session_id}/memory/working/finish", json={"promote": {}}
    )

    assert missing.status_code == 404
    assert foreign.status_code == 422
    assert off.status_code == 409 and "longterm" in off.json()["detail"]
    # A refused promotion leaves the task open; only the last call closes it.
    assert emptied.status_code == 200
    assert emptied.json()["context"]["working"]["items"] == {}
    assert emptied.json()["memory"]["working"]["versions"][-1]["origin"] == "manual"


def test_a_new_session_of_the_same_scope_starts_with_its_long_term_memory(client):
    spec = _memory_spec(longterm_enabled=True, longterm_write="manual")
    first = client.post("/api/sessions", json=spec).json()["id"]
    client.post(
        f"/api/sessions/{first}/memory/longterm/items",
        json={"key": "name", "value": "Сергей", "kind": "profile"},
    )

    same_scope = client.get(f"/api/sessions/{client.post('/api/sessions', json=spec).json()['id']}").json()
    other = _memory_spec(longterm_enabled=True, longterm_scope="work")
    other_scope = client.get(
        f"/api/sessions/{client.post('/api/sessions', json=other).json()['id']}"
    ).json()

    # The long-term layer is keyed by the space, so a session that never saw
    # the dialogue starts out knowing what the space holds.
    assert same_scope["context"]["longterm"]["items"]["name"]["value"] == "Сергей"
    assert same_scope["memory"]["longterm"]["sessions"] == 2
    assert other_scope["context"]["longterm"] is None
    assert other_scope["memory"]["longterm"]["sessions"] == 1


def test_a_branch_copies_the_working_layer_and_shares_the_scope(client):
    client.app.state.providers["zen"].client = StubLLM(script=[("Ответ.", [])])
    session_id = client.post(
        "/api/sessions",
        json=_memory_spec(
            working_enabled=True, longterm_enabled=True, longterm_write="manual"
        ),
    ).json()["id"]
    sse_events(
        client.post(f"/api/sessions/{session_id}/messages", json={"text": "цель — макет"})
    )
    client.post(
        f"/api/sessions/{session_id}/memory/working/items",
        json={"key": "goal", "value": "макет к среде", "kind": "goal"},
    )
    client.post(
        f"/api/sessions/{session_id}/memory/longterm/items",
        json={"key": "name", "value": "Сергей", "kind": "profile"},
    )

    branch_id = client.post(
        f"/api/sessions/{session_id}/fork", json={"seq": 2}
    ).json()["id"]
    client.post(
        f"/api/sessions/{session_id}/memory/working/items",
        json={"key": "blocker", "value": "нет доступа", "kind": "blocker"},
    )
    branch = client.get(f"/api/sessions/{branch_id}").json()

    # The task data is part of the snapshot; the space is shared, not copied.
    assert branch["context"]["working"]["items"]["goal"]["value"] == "макет к среде"
    assert "blocker" not in branch["context"]["working"]["items"]
    assert branch["context"]["longterm"]["items"]["name"]["value"] == "Сергей"
    assert branch["memory"]["longterm"]["sessions"] == 2


def test_a_layer_the_spec_keeps_off_refuses_manual_writes(client):
    session_id = client.post(
        "/api/sessions", json={"spec": {"memory": {"working_enabled": True}}}
    ).json()["id"]

    refused = client.put(
        f"/api/sessions/{session_id}/memory/longterm",
        json={"items": {"name": {"value": "Сергей", "kind": "profile"}}},
    )
    unknown = client.put(
        f"/api/sessions/{session_id}/memory/episodic", json={"items": {}}
    )

    assert refused.status_code == 409
    assert unknown.status_code == 404


def test_memory_update_failure_keeps_the_completed_chat_turn(client):
    client.app.state.providers["zen"].client = StubLLM(
        script=[("Ответ сохранён.", []), ("не json", [])]
    )
    session_id = client.post(
        "/api/sessions", json={"spec": {"memory": {"working_enabled": True}}}
    ).json()["id"]

    events = sse_events(
        client.post(f"/api/sessions/{session_id}/messages", json={"text": "вопрос"})
    )
    detail = client.get(f"/api/sessions/{session_id}").json()

    assert [event["type"] for event in events[-2:]] == ["memory_error", "done"]
    assert [message["role"] for message in detail["messages"]] == ["user", "assistant"]
    assert [row["kind"] for row in detail["usage"]] == ["chat", "memory"]
    assert detail["memory"]["working"]["versions"] == []


def test_failed_call_rolls_back_the_question(client):
    session_id = client.post("/api/sessions", json={}).json()["id"]
    client.app.state.providers["zen"].client = StubLLM(fail_with="Zen недоступен")

    response = client.post(
        f"/api/sessions/{session_id}/messages", json={"text": "привет"}
    )
    assert sse_events(response)[-1]["type"] == "error"
    assert client.get(f"/api/sessions/{session_id}").json()["messages"] == []


def test_missing_key_answers_503(client):
    session_id = client.post("/api/sessions", json={}).json()["id"]
    client.app.state.providers["zen"].client = None

    response = client.post(
        f"/api/sessions/{session_id}/messages", json={"text": "привет"}
    )
    assert response.status_code == 503


def test_session_resolves_client_from_spec_provider(client):
    other = StubLLM("Другой провайдер")
    client.app.state.providers = ProviderRegistry(
        {
            "other": Provider(
                name="other",
                client=other,
                catalog=Catalog(),
                capabilities=ZEN_CAPABILITIES,
                credential_env="OTHER_API_KEY",
                base_url="https://other.example",
            )
        }
    )
    session = client.post(
        "/api/sessions",
        json={"spec": {"model": {"provider": "other", "model": "gpt-5-nano"}}},
    ).json()

    response = client.post(
        f"/api/sessions/{session['id']}/messages", json={"text": "привет"}
    )

    assert response.status_code == 200
    assert other.requests[-1].model == "gpt-5-nano"


def test_schema_endpoint_is_the_form_contract(client):
    schema = client.get("/api/agent-spec-schema").json()

    assert schema["$defs"]["ReasoningSection"]["properties"]["strategy"]["enum"] == [
        "direct", "step_by_step", "self_prompt", "experts"
    ]
    assert schema["$defs"]["OutputSection"]["properties"]["mode"]["enum"] == [
        "free", "api", "prompt"
    ]
    assert set(schema["properties"]) == {
        "model", "sampling", "output", "reasoning", "context", "memory", "tools",
        "task", "judge"
    }
    assert schema["properties"]["context"]["discriminator"]["propertyName"] == "mode"
    assert client.get("/api/models").json()[0]["provider"] == "zen"


def test_zero_budget_means_no_limit_for_sessions_and_runs(client):
    session = client.post("/api/sessions", json={"spec": {"budget_usd": 0}})
    assert session.status_code == 200
    assert client.get(f"/api/sessions/{session.json()['id']}").json()["spec"]["context"]["budget_usd"] is None

    estimate = client.post(
        "/api/runs/estimate",
        json={"prompt": "проверка", "budget_usd": 0, "specs": [{}]},
    )
    assert estimate.status_code == 200
    quoted = client.app.state.run_quotes[estimate.json()["quote_id"]]
    assert quoted["request"].budget_usd is None


def test_session_listing_and_deletion(client):
    session_id = client.post("/api/sessions", json={"title": "х"}).json()["id"]
    assert len(client.get("/api/sessions").json()) == 1

    assert client.delete(f"/api/sessions/{session_id}").status_code == 200
    assert client.get("/api/sessions").json() == []
    assert client.get(f"/api/sessions/{session_id}").status_code == 404


def test_legacy_null_output_limit_uses_current_default(client):
    session_id = client.post("/api/sessions", json={}).json()["id"]
    conn = client.app.state.db
    row = conn.execute(
        "SELECT a.hash, a.spec_json FROM sessions s "
        "JOIN agent_specs a ON a.hash = s.spec_hash WHERE s.id = ?",
        (session_id,),
    ).fetchone()
    legacy_spec = {
        "name": "legacy",
        "model": "gpt-5-nano",
        "role": "роль",
        "max_output_tokens": None,
    }
    conn.execute(
        "UPDATE agent_specs SET spec_json = ? WHERE hash = ?",
        (json.dumps(legacy_spec), row["hash"]),
    )
    conn.commit()

    detail = client.get(f"/api/sessions/{session_id}")

    assert detail.status_code == 200
    assert detail.json()["spec"]["output"]["max_output_tokens"] == 200
    assert client.get("/api/sessions").json()[0]["spec"]["output"]["max_output_tokens"] == 200


def test_new_session_rejects_null_output_limit(client):
    response = client.post(
        "/api/sessions", json={"spec": {"output": {"max_output_tokens": None}}}
    )

    assert response.status_code == 422


def test_presets_are_versioned_pointers_and_supply_new_sessions(client):
    created = client.post(
        "/api/presets",
        json={"name": "быстрый", "spec": {"model": {"role": "version-1"}}, "is_default": True},
    )
    assert created.status_code == 201
    first = created.json()
    assert first["is_default"] is True
    assert client.get(f"/api/presets/{first['id']}").json() == first
    assert client.post(
        "/api/presets", json={"name": "быстрый", "spec": {}}
    ).status_code == 409

    old_session = client.post("/api/sessions", json={}).json()
    assert old_session["preset_id"] == first["id"]
    assert old_session["spec"]["model"]["role"] == "version-1"

    updated = client.put(
        f"/api/presets/{first['id']}",
        json={"name": "быстрый", "spec": {"model": {"role": "version-2"}}},
    )
    assert updated.status_code == 200
    assert updated.json()["spec_hash"] != first["spec_hash"]

    new_session = client.post(
        "/api/sessions", json={"preset_id": first["id"]}
    ).json()
    assert new_session["spec"]["model"]["role"] == "version-2"
    assert client.get(f"/api/sessions/{old_session['id']}").json()["spec"]["model"]["role"] == "version-1"

    second = client.post(
        "/api/presets", json={"name": "точный", "spec": {"model": {"role": "precise"}}}
    ).json()
    made_default = client.put(f"/api/presets/{second['id']}/default")
    assert made_default.status_code == 200
    presets = client.get("/api/presets").json()
    assert [preset["id"] for preset in presets if preset["is_default"]] == [second["id"]]
    assert client.post("/api/sessions", json={}).json()["preset_id"] == second["id"]

    assert client.delete(f"/api/presets/{first['id']}").status_code == 200
    assert client.get(f"/api/presets/{first['id']}").status_code == 404
    assert client.get(f"/api/sessions/{old_session['id']}").status_code == 200


def test_patch_repoints_spec_without_changing_history_or_old_ledger(client):
    session_id = client.post("/api/sessions", json={}).json()["id"]
    client.post(f"/api/sessions/{session_id}/messages", json={"text": "до правки"})
    before = client.get(f"/api/sessions/{session_id}").json()
    assert before["editable"] is True
    old_hash = before["usage"][0]["spec_hash"]
    updated_spec = before["spec"]
    updated_spec["model"]["role"] = "Новая роль"

    response = client.patch(f"/api/sessions/{session_id}/spec", json=updated_spec)

    assert response.status_code == 200
    after = response.json()
    assert after["messages"] == before["messages"]
    assert after["spec"]["model"]["role"] == "Новая роль"
    current_hash = client.app.state.db.execute(
        "SELECT spec_hash FROM sessions WHERE id = ?", (session_id,)
    ).fetchone()["spec_hash"]
    assert current_hash != old_hash
    assert after["usage"][0]["spec_hash"] == old_hash
    client.post(f"/api/sessions/{session_id}/messages", json={"text": "после правки"})
    usage_hashes = [
        row["spec_hash"]
        for row in client.get(f"/api/sessions/{session_id}").json()["usage"]
    ]
    assert usage_hashes == [old_hash, current_hash]


def test_patch_rejects_matrix_session(client):
    spec = AgentSpec()
    run_id = repo.create_run(
        client.app.state.db,
        name="matrix",
        prompt="задача",
        repeats=1,
        budget_usd=None,
        estimated_cost_usd=0,
        specs=[spec],
    )
    session_id = repo.create_session(client.app.state.db, spec)
    client.app.state.db.execute(
        "UPDATE sessions SET run_id = ? WHERE id = ?", (run_id, session_id)
    )
    client.app.state.db.commit()

    response = client.patch(
        f"/api/sessions/{session_id}/spec", json=spec.model_dump(mode="json")
    )

    assert response.status_code == 409
    assert client.get(f"/api/sessions/{session_id}").json()["editable"] is False


def test_session_rejects_ambiguous_spec_source(client):
    preset = client.post(
        "/api/presets", json={"name": "one", "spec": {}}
    ).json()
    response = client.post(
        "/api/sessions", json={"preset_id": preset["id"], "spec": {}}
    )
    assert response.status_code == 422


def test_workspace_must_exist(client):
    response = client.post("/api/sessions", json={"workspace_path": "/nope/nowhere"})
    assert response.status_code == 400
    assert "Папка не найдена" in response.json()["detail"]


def test_checkpoint_and_fork_copy_only_state_available_at_seq(client):
    spec = AgentSpec(
        context={"mode": "window", "keep_last_messages": 2},
        memory={"working_enabled": True},
    )
    session_id = repo.create_session(client.app.state.db, spec, "исходная")
    first_id = repo.append_message(client.app.state.db, session_id, "user", "вопрос")
    answer_id = repo.append_message(client.app.state.db, session_id, "assistant", "ответ")
    repo.record_tool_call(
        client.app.state.db,
        session_id,
        ToolCall(id="original-call", tool="read_file", arguments_json='{"path":"a.txt"}'),
        ToolResult(call_id="original-call", output="alpha", duration_ms=7),
        "ok",
    )
    repo.append_message(client.app.state.db, session_id, "assistant", "после инструмента")
    repo.append_message(client.app.state.db, session_id, "user", "поздний вопрос")
    repo.save_summary(
        client.app.state.db,
        session_id,
        covered_seq=2,
        message_count=2,
        text="ранняя сводка",
        tokens=4,
    )
    repo.save_summary(
        client.app.state.db,
        session_id,
        covered_seq=4,
        message_count=4,
        text="поздняя сводка",
        tokens=6,
    )
    for covered_seq, stage in ((2, "early"), (4, "late")):
        repo.save_memory(
            client.app.state.db,
            "working",
            session_id=session_id,
            covered_seq=covered_seq,
            items={"stage": {"value": stage, "kind": "step", "origin": "router", "seq": covered_seq}},
            tokens=3,
            origin="router",
        )
    repo.record_usage(
        client.app.state.db,
        session_id,
        spec.model.model,
        Usage(tokens_in=10, tokens_out=2, cost=0.01),
        spec_hash=spec.hash,
        message_id=answer_id,
    )
    task = repo.create_task(client.app.state.db, session_id)
    repo.save_task_plan(client.app.state.db, task["id"], "1. поздний шаг")

    checkpoint = client.post(
        f"/api/sessions/{session_id}/checkpoints",
        json={"seq": 2, "name": "развилка"},
    )
    assert checkpoint.status_code == 200
    assert checkpoint.json()["checkpoints"] == [
        {"session_id": session_id, "seq": 2, "name": "развилка"}
    ]

    response = client.post(f"/api/sessions/{session_id}/fork", json={"seq": 3})

    assert response.status_code == 200
    branch = response.json()
    assert branch["parent_id"] == session_id
    assert branch["fork_seq"] == 3
    assert branch["title"] == "ветка от #3"
    # The task was started after the fork point, so the branch has none.
    assert branch["task"] is None
    assert branch["spec_hash"] == spec.hash
    assert [item["role"] for item in branch["messages"]] == [
        "user",
        "assistant",
        "tool",
    ]
    assert [item["seq"] for item in branch["messages"]] == [1, 2, 3]
    assert branch["messages"][0]["id"] != first_id
    assert branch["messages"][2]["tool"] == "read_file"
    assert branch["messages"][2]["arguments_json"] == '{"path":"a.txt"}'
    assert branch["messages"][2]["content"] == "alpha"
    assert [item["summary"] for item in branch["summaries"]] == ["ранняя сводка"]
    assert [
        item["items"]["stage"]["value"]
        for item in branch["memory"]["working"]["versions"]
    ] == ["early"]
    assert branch["checkpoints"] == []
    assert branch["usage"] == []
    assert branch["totals"] == {
        "requests": 0,
        "tokens_in": 0,
        "tokens_out": 0,
        "reasoning": 0,
        "cost_usd": 0,
    }

    sessions = client.get("/api/sessions").json()
    branch_summary = next(item for item in sessions if item["id"] == branch["id"])
    parent_summary = next(item for item in sessions if item["id"] == session_id)
    assert (branch_summary["parent_id"], branch_summary["fork_seq"]) == (session_id, 3)
    assert branch_summary["cost_usd"] == 0
    assert parent_summary["cost_usd"] == 0.01

    nested = client.post(
        f"/api/sessions/{branch['id']}/fork",
        json={"seq": 2, "title": "альтернатива"},
    ).json()
    assert nested["parent_id"] == branch["id"]
    assert nested["title"] == "альтернатива"


def test_fork_accepts_saved_spec_without_rerouting_memory(client):
    db = client.app.state.db
    original = AgentSpec(memory={"working_enabled": False})
    source_id = repo.create_session(db, original, "source")
    repo.append_message(db, source_id, "user", "remember this")
    repo.append_message(db, source_id, "assistant", "previous answer")
    replay = AgentSpec(
        model={"role": "saved role"},
        memory={"working_enabled": True, "longterm_enabled": True, "longterm_scope": "replay-scope"},
    )

    response = client.post(f"/api/sessions/{source_id}/fork", json={"seq": 2, "spec": replay.model_dump()})

    assert response.status_code == 200
    branch = response.json()
    assert branch["spec"] == replay.model_dump()
    assert branch["memory"]["longterm"]["scope"] == "replay-scope"
    assert branch["parent_id"] == source_id
    assert len(branch["messages"]) == 2
    assert branch["usage"] == []
    assert client.app.state.providers["zen"].client.requests == []
    assert client.get(f"/api/sessions/{source_id}").json()["spec"] == original.model_dump()


def test_checkpoint_and_fork_reject_missing_message(client):
    session_id = client.post("/api/sessions", json={}).json()["id"]

    assert client.post(
        f"/api/sessions/{session_id}/checkpoints", json={"seq": 1}
    ).status_code == 400
    assert client.post(
        f"/api/sessions/{session_id}/fork", json={"seq": 1}
    ).status_code == 400
    assert client.post(
        "/api/sessions/missing/fork", json={"seq": 1}
    ).status_code == 404


def test_tool_calls_are_stored_and_restored(client, tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "notes.md").write_text("alpha\n", encoding="utf-8")

    client.app.state.providers["zen"].client = StubLLM(
        script=[
            (
                "",
                [
                    ToolCall(
                        id="c1", tool="read_file", arguments_json='{"path": "notes.md"}'
                    )
                ],
            ),
            ("Там alpha.", []),
        ]
    )
    session_id = client.post(
        "/api/sessions",
        json={"workspace_path": str(workspace), "spec": {"approval_mode": "auto"}},
    ).json()["id"]

    events = sse_events(
        client.post(f"/api/sessions/{session_id}/messages", json={"text": "что там?"})
    )
    kinds = [event["type"] for event in events]
    assert "tool" in kinds and "tool_result" in kinds

    detail = client.get(f"/api/sessions/{session_id}").json()
    assert detail["workspace_path"] == str(workspace)
    stored = [m for m in detail["messages"] if m["role"] == "tool"]
    assert len(stored) == 1
    assert stored[0]["tool"] == "read_file"
    assert stored[0]["status"] == "ok"
    assert stored[0]["content"] == "alpha\n"
    # The transcript keeps its order: question, call, answer.
    assert [m["role"] for m in detail["messages"]] == ["user", "tool", "assistant"]


def test_approval_endpoint_resolves_a_pending_call(client):
    """The port parks on a future; the second HTTP call is what wakes it."""

    async def park():
        pending = client.app.state.approvals
        task = asyncio.create_task(
            HttpApprovals(pending, "s1").confirm(
                ToolCall(id="c1", tool="write_file"), "diff"
            )
        )
        while "s1:c1" not in pending:
            await asyncio.sleep(0)
        return task

    task = client.portal.call(park)
    answer = client.post("/api/sessions/s1/approvals/c1", json={"approved": True})
    assert answer.json() == {"call_id": "c1", "approved": True}
    assert client.portal.call(_await, task) is True


def test_approval_endpoint_404s_when_nothing_waits(client):
    response = client.post("/api/sessions/s1/approvals/nope", json={"approved": True})
    assert response.status_code == 404


def test_matrix_cost_requires_confirmation_then_returns_comparison(client):
    quote = client.post(
        "/api/runs/estimate",
        json={"prompt": "проверь", "repeats": 1, "specs": [{"name": "nano"}]},
    )
    assert quote.status_code == 200
    quote_body = quote.json()
    assert quote_body["quote"]["agent_calls"] == 1
    assert quote_body["quote"]["judge_calls"] == 1

    client.app.state.providers["zen"].client = StubLLM(
        script=[
            ("ответ", []),
            (json.dumps({"score": 8, "rationale": "достаточно"}), []),
        ]
    )
    response = client.post("/api/runs", json={"quote_id": quote_body["quote_id"]})

    assert response.status_code == 200
    dashboard = response.json()
    assert len(dashboard["agents"]) == 1
    for _ in range(20):
        report = client.get(f"/api/runs/{dashboard['run_id']}").json()
        if report["status"] != "running":
            break
        time.sleep(0.01)
    assert report["status"] == "completed"
    assert report["comparison"][0]["passed"] == 1
    assert report["totals"]["requests"] == 2
    live = client.get(f"/api/runs/{dashboard['run_id']}/dashboard").json()
    assert live["active"] is False
    assert live["agents"][0]["state"] == "completed"
    assert live["provider"]["limit"] > 0
    # A quote cannot be replayed to trigger a second paid run.
    assert client.post("/api/runs", json={"quote_id": quote_body["quote_id"]}).status_code == 409

    exported_json = client.get(f"/api/runs/{dashboard['run_id']}/export?format=json")
    assert exported_json.status_code == 200
    assert exported_json.headers["content-type"].startswith("application/json")
    assert exported_json.json()["comparison"][0]["name"] == "Конфигурация 1"

    exported_csv = client.get(f"/api/runs/{dashboard['run_id']}/export?format=csv")
    assert exported_csv.status_code == 200
    assert "config_index,repeat_index" in exported_csv.text


def test_matrix_quote_resolves_preset_to_an_immutable_spec(client):
    preset = client.post(
        "/api/presets", json={"name": "matrix", "spec": {"model": {"role": "before"}}}
    ).json()
    quote = client.post(
        "/api/runs/estimate",
        json={"prompt": "проверь", "repeats": 1, "preset_ids": [preset["id"]]},
    )
    assert quote.status_code == 200
    quoted = client.app.state.run_quotes[quote.json()["quote_id"]]

    client.put(
        f"/api/presets/{preset['id']}",
        json={"name": "matrix", "spec": {"model": {"role": "after"}}},
    )

    assert quoted["specs"][0].model.role == "before"


def test_matrix_requires_exactly_one_configuration_source(client):
    assert client.post(
        "/api/runs/estimate", json={"prompt": "проверь"}
    ).status_code == 422
    assert client.post(
        "/api/runs/estimate",
        json={"prompt": "проверь", "specs": [{}], "preset_ids": ["x"]},
    ).status_code == 422


def test_session_keeps_its_selected_executor(client, tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    created = client.post(
        "/api/sessions",
        json={"workspace_path": str(workspace), "executor_kind": "docker"},
    )

    assert created.status_code == 200
    assert created.json()["executor_kind"] == "docker"
    assert client.get(f"/api/sessions/{created.json()['id']}").json()["executor_kind"] == "docker"


def test_the_planning_stage_plans_and_only_execution_gets_tools(client, tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    client.app.state.providers["zen"].client = StubLLM(
        script=[
            ("1. Проверить требования\n2. Выполнить задачу\nПроверка:\n- файл на месте", []),
            ("", [ToolCall(id="c1", tool="write_file", arguments_json='{"path":"done.txt","content":"ok"}')]),
            ("Готово.", []),
        ]
    )
    session_id = client.post(
        "/api/sessions",
        json={"workspace_path": str(workspace), "spec": {"approval_mode": "auto"}},
    ).json()["id"]

    started = client.post(f"/api/sessions/{session_id}/task")
    assert started.status_code == 200
    assert started.json()["task"]["stage"] == "planning"
    assert started.json()["task"]["expected_action"] == {
        "actor": "agent",
        "text": "составить план",
    }
    assert client.post(f"/api/sessions/{session_id}/task").status_code == 409
    refused = task_transition(client, session_id, "execution")
    assert refused.status_code == 409
    assert "не разобран" in refused.json()["detail"]["message"]

    response = client.post(
        f"/api/sessions/{session_id}/messages", json={"text": "Составь план"}
    )
    assert [event["type"] for event in sse_events(response)][-2] == "task"
    planned = client.get(f"/api/sessions/{session_id}").json()["task"]
    assert [step["title"] for step in planned["steps"]] == [
        "Проверить требования",
        "Выполнить задачу",
    ]
    assert planned["criteria"] == ["файл на месте"]
    assert client.app.state.providers["zen"].client.requests[0].tools == []
    assert not (workspace / "done.txt").exists()

    executing = task_transition(client, session_id, "execution")
    assert executing.status_code == 200
    assert executing.json()["task"]["stage"] == "execution"
    assert executing.json()["task"]["current_step"] == 1

    client.post(f"/api/sessions/{session_id}/messages", json={"text": "Выполни план"})
    assert {tool.name for tool in client.app.state.providers["zen"].client.requests[1].tools} == {
        "read_file", "write_file", "edit_file", "list_dir", "grep", "run_command"
    }
    assert (workspace / "done.txt").read_text(encoding="utf-8") == "ok"


def test_validation_reads_and_runs_but_never_writes(client, tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    client.app.state.providers["zen"].client = StubLLM("Проверил.")
    session_id = client.post(
        "/api/sessions",
        json={"workspace_path": str(workspace), "spec": {"approval_mode": "auto"}},
    ).json()["id"]
    client.post(f"/api/sessions/{session_id}/task")
    task_id = client.get(f"/api/sessions/{session_id}/task").json()["id"]
    repo.save_task_plan(client.app.state.db, task_id, "1. Сделать\nПроверка:\n- сделано")
    task_transition(client, session_id, "execution")
    task_step(client, session_id, 1, status="done")
    task_transition(client, session_id, "validation")

    client.post(f"/api/sessions/{session_id}/messages", json={"text": "Проверь"})

    offered = {tool.name for tool in client.app.state.providers["zen"].client.requests[-1].tools}
    assert offered == {"read_file", "list_dir", "grep"}


def test_cancelling_a_task_frees_the_session_and_closes_the_working_layer(client):
    session_id = client.post("/api/sessions", json=_memory_spec(working_enabled=True)).json()["id"]
    client.post(f"/api/sessions/{session_id}/task")
    repo.save_memory(
        client.app.state.db,
        "working",
        session_id=session_id,
        covered_seq=0,
        items={"goal": {"value": "собрать отчёт", "kind": "goal", "origin": "manual", "seq": 0}},
        tokens=4,
        origin="manual",
    )

    cancelled = task_transition(client, session_id, "cancelled")

    assert cancelled.status_code == 200
    assert cancelled.json()["task"] is None
    versions = cancelled.json()["memory"]["working"]["versions"]
    assert versions[-1]["items"] == {} and len(versions) == 2
    assert client.post(f"/api/sessions/{session_id}/task").status_code == 200
    # The strip loses a closed task, the transcript does not lose its marks.
    journal = client.get(f"/api/sessions/{session_id}").json()["task_journal"]
    assert [entry["kind"] for entry in journal] == ["start", "stage", "start"]
    assert journal[1]["to_stage"] == "cancelled"


def test_a_paused_task_keeps_its_stage_and_loses_its_tools(client, tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    client.app.state.providers["zen"].client = StubLLM("Подожду.")
    session_id = client.post(
        "/api/sessions",
        json={"workspace_path": str(workspace), "spec": {"approval_mode": "auto"}},
    ).json()["id"]
    client.post(f"/api/sessions/{session_id}/task")
    task_id = client.get(f"/api/sessions/{session_id}/task").json()["id"]
    repo.save_task_plan(client.app.state.db, task_id, "1. Написать a.txt\nПроверка:\n- файл есть")
    task_transition(client, session_id, "execution")

    paused = task_pause(client, session_id, True)
    assert paused.json()["task"]["paused"] is True
    assert paused.json()["task"]["available"] == [
        {"stage": "cancelled", "needs_force": False, "reason": None, "allowed": True}
    ]
    assert task_pause(client, session_id, True).status_code == 409

    client.post(f"/api/sessions/{session_id}/messages", json={"text": "Что там?"})
    assert client.app.state.providers["zen"].client.requests[-1].tools == []

    resumed = task_pause(client, session_id, False)
    assert resumed.json()["task"]["stage"] == "execution"
    assert resumed.json()["task"]["paused"] is False
    assert resumed.json()["task"]["expected_action"]["text"] == (
        "выполнить шаг 1 «Написать a.txt»"
    )


def test_a_step_is_edited_by_hand_and_an_unknown_one_is_refused(client):
    session_id = client.post("/api/sessions", json={}).json()["id"]
    client.post(f"/api/sessions/{session_id}/task")
    task_id = client.get(f"/api/sessions/{session_id}/task").json()["id"]
    repo.save_task_plan(client.app.state.db, task_id, "1. Первый\n2. Второй")
    task_transition(client, session_id, "execution")

    edited = task_step(client, session_id, 1, status="blocked", note="нет доступа")

    assert edited.status_code == 200
    assert edited.json()["task"]["steps"][0]["status"] == "blocked"
    assert "снять блокер шага 1" in edited.json()["task"]["expected_action"]["text"]
    assert task_step(client, session_id, 9, status="done").status_code == 404


def test_the_task_state_survives_an_application_restart(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTBOX_DB", str(tmp_path / "restart.db"))
    first_app = create_app()
    with TestClient(first_app) as first_client:
        first_app.state.providers["zen"].client = StubLLM(
            "1. Написать отчёт\n2. Отправить\nПроверка:\n- отчёт отправлен"
        )
        session_id = first_client.post("/api/sessions", json={}).json()["id"]
        first_client.post(f"/api/sessions/{session_id}/task")
        first_client.post(
            f"/api/sessions/{session_id}/messages", json={"text": "Составь план"}
        )
        task_transition(first_client, session_id, "execution")
        task_pause(first_client, session_id, True)

    restarted_app = create_app()
    with TestClient(restarted_app) as restarted_client:
        restored = restarted_client.get(f"/api/sessions/{session_id}").json()

    assert restored["task"]["stage"] == "execution"
    assert restored["task"]["paused"] is True
    assert [step["title"] for step in restored["task"]["steps"]] == [
        "Написать отчёт",
        "Отправить",
    ]
    # The block is in the payload, so "продолжай" needs no re-explanation.
    block = restored["context"]["task"]
    assert block["stage"] == "execution" and block["payload_tokens"] > 0
    assert "Написать отчёт" in block["content"]
    assert "(на паузе)" in block["content"]
    assert block["content"].endswith("продолжить задачу")


def test_task_transition_rejects_force_and_stale_parallel_clicks(client):
    session_id = client.post("/api/sessions", json={}).json()["id"]
    client.post(f"/api/sessions/{session_id}/task")
    task = client.get(f"/api/sessions/{session_id}/task").json()
    repo.save_task_plan(client.app.state.db, task["id"], "1. Сделать\nПроверка:\n- готово")
    detail = client.get(f"/api/sessions/{session_id}").json()
    payload = {
        "stage": "execution",
        "expected_revision": detail["task"]["revision"],
        "expected_invariant_hash": detail["invariants"]["version_hash"],
    }

    forced = client.post(
        f"/api/sessions/{session_id}/task/transition",
        json={**payload, "force": True},
    )
    first = client.post(f"/api/sessions/{session_id}/task/transition", json=payload)
    stale = client.post(f"/api/sessions/{session_id}/task/transition", json=payload)

    assert forced.status_code == 409
    assert forced.json()["detail"]["code"] == "force_removed"
    assert first.status_code == 200 and first.json()["task"]["stage"] == "execution"
    assert stale.status_code == 409
    assert stale.json()["detail"]["code"] == "stale_task"
    journal = client.get(f"/api/sessions/{session_id}").json()["task_journal"]
    assert [item["kind"] for item in journal][-3:] == ["rejected", "stage", "rejected"]


def test_plan_approval_is_bound_to_plan_and_invariant_versions(client):
    session_id = client.post("/api/sessions", json={}).json()["id"]
    client.post(f"/api/sessions/{session_id}/task")
    task = client.get(f"/api/sessions/{session_id}/task").json()
    repo.save_task_plan(client.app.state.db, task["id"], "1. Сделать\nПроверка:\n- готово")

    approved = task_transition(client, session_id, "execution").json()
    task = approved["task"]

    assert task["approved_plan_hash"]
    assert task["approved_invariant_hash"] == approved["invariants"]["version_hash"]
    assert task["revision"] > 0


class RecordingExecutor:
    def __init__(self):
        self.calls = []

    def describe(self):
        return []

    def preview(self, call):
        return call.tool

    async def run(self, call):
        self.calls.append(call)
        return ToolResult(call_id=call.id, output="ran")


async def test_a_tool_waiting_during_pause_cannot_write_late(client):
    session_id = client.post("/api/sessions", json={}).json()["id"]
    client.post(f"/api/sessions/{session_id}/task")
    task = client.get(f"/api/sessions/{session_id}/task").json()
    repo.save_task_plan(client.app.state.db, task["id"], "1. Сделать")
    task_transition(client, session_id, "execution")
    snapshot = client.get(f"/api/sessions/{session_id}").json()["task"]
    recorder = RecordingExecutor()
    guarded = TaskBoundExecutor(recorder, client.app.state.db, snapshot)
    task_pause(client, session_id, True)

    result = await guarded.run(ToolCall(id="late", tool="write_file"))

    assert result.ok is False
    assert "изменилось" in result.output
    assert recorder.calls == []


async def _await(task):
    return await task


def _memory_spec(**memory) -> dict:
    return {
        "spec": {
            "context": {"mode": "window", "keep_last_messages": 2},
            "memory": {"max_tokens": 64, **memory},
        }
    }


def test_confirm_parks_longterm_candidates_until_a_human_accepts_one(client):
    client.app.state.providers["zen"].client = StubLLM(
        script=[
            ("Запомнил.", []),
            (
                '{"longterm":{"name":{"value":"Сергей","kind":"profile"},'
                '"stack":{"value":"Kotlin","kind":"decision"}}}',
                [],
            ),
        ]
    )
    session_id = client.post(
        "/api/sessions",
        json=_memory_spec(longterm_enabled=True, longterm_write="confirm"),
    ).json()["id"]

    events = sse_events(
        client.post(
            f"/api/sessions/{session_id}/messages", json={"text": "Меня зовут Сергей."}
        )
    )

    # Memory that outlives the dialogue is not written by the router alone.
    assert not [event for event in events if event["type"] == "memory"]
    candidates = next(event for event in events if event["type"] == "memory_proposals")
    assert {item["key"] for item in candidates["proposals"]} == {"name", "stack"}
    detail = client.get(f"/api/sessions/{session_id}").json()
    assert detail["memory"]["longterm"]["versions"] == []
    assert detail["context"]["longterm"] is None

    proposals = {item["key"]: item["id"] for item in detail["memory"]["longterm"]["proposals"]}
    accepted = client.post(
        f"/api/sessions/{session_id}/memory/proposals/{proposals['name']}",
        json={"decision": "accept"},
    )
    rejected = client.post(
        f"/api/sessions/{session_id}/memory/proposals/{proposals['stack']}",
        json={"decision": "reject"},
    )

    assert accepted.status_code == 200 and rejected.status_code == 200
    detail = rejected.json()
    assert detail["memory"]["longterm"]["proposals"] == []
    version = detail["memory"]["longterm"]["versions"][-1]
    # The router produced the item, a human wrote the version: both are visible.
    assert version["origin"] == "manual"
    assert version["items"] == {
        "name": {"value": "Сергей", "kind": "profile", "origin": "router", "seq": 2}
    }
    assert detail["context"]["longterm"]["items"]["name"]["value"] == "Сергей"


def test_auto_writes_longterm_at_once_and_manual_never_asks_the_router(client):
    client.app.state.providers["zen"].client = StubLLM(
        script=[
            ("Запомнил.", []),
            ('{"longterm":{"name":{"value":"Сергей","kind":"profile"}}}', []),
        ]
    )
    session_id = client.post(
        "/api/sessions",
        json=_memory_spec(longterm_enabled=True, longterm_write="auto"),
    ).json()["id"]

    events = sse_events(
        client.post(
            f"/api/sessions/{session_id}/messages", json={"text": "Меня зовут Сергей."}
        )
    )

    stored = next(event for event in events if event["type"] == "memory")
    assert stored["layer"] == "longterm" and stored["origin"] == "router"
    assert stored["items"]["name"]["value"] == "Сергей"

    client.app.state.providers["zen"].client = StubLLM(script=[("Ага.", [])])
    manual_id = client.post(
        "/api/sessions",
        json=_memory_spec(longterm_enabled=True, longterm_write="manual"),
    ).json()["id"]
    sse_events(
        client.post(f"/api/sessions/{manual_id}/messages", json={"text": "Привет."})
    )

    # A layer filled by hand costs no routing call at all.
    assert len(client.app.state.providers["zen"].client.requests) == 1
    manual = client.get(f"/api/sessions/{manual_id}").json()
    assert [row["kind"] for row in manual["usage"]] == ["chat"]


def test_router_reads_each_message_once_and_reports_refused_kinds(client):
    client.app.state.providers["zen"].client = StubLLM(
        script=[
            ("Запомнил.", []),
            ('{"working":{"goal":{"value":"собрать ТЗ","kind":"goal"}}}', []),
            ("Ага.", []),
            ('{"working":{"name":{"value":"Сергей","kind":"profile"}}}', []),
        ]
    )
    session_id = client.post(
        "/api/sessions", json=_memory_spec(working_enabled=True)
    ).json()["id"]

    sse_events(
        client.post(f"/api/sessions/{session_id}/messages", json={"text": "Нужно ТЗ."})
    )
    second = sse_events(
        client.post(f"/api/sessions/{session_id}/messages", json={"text": "Меня зовут Сергей."})
    )

    prompt = client.app.state.providers["zen"].client.requests[-1].messages[1]["content"]
    assert "Нужно ТЗ" not in prompt and "Меня зовут Сергей" in prompt
    refused = next(event for event in second if event["type"] == "memory_rejected")
    assert refused["keys"] == ["working.name"]
    # Nothing was written under a foreign kind, and the turn still answered.
    detail = client.get(f"/api/sessions/{session_id}").json()
    assert sorted(detail["context"]["working"]["items"]) == ["goal"]
    assert second[-1]["type"] == "done"
