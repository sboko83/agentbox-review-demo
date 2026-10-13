import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from agentbox.api.app import create_app
from agentbox.core.invariants import InvariantRule, InvariantSet
from agentbox.core.ports import Usage
from agentbox.core.spec import AgentSpec
from agentbox.store import repository as repo
from agentbox.store.db import init_db


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTBOX_DB", str(tmp_path / "test.db"))
    with TestClient(create_app()) as test_client:
        yield test_client


def rules(text: str = "Python 3.9+") -> list[dict]:
    return [{"id": "stack.python", "category": "stack", "text": text}]


def test_hash_is_canonical_and_rule_ids_are_unique():
    first = InvariantSet(
        rules=[
            InvariantRule(id="stack.python", category="stack", text="Python 3.9+"),
            InvariantRule(id="arch.api", category="architecture", text="HTTP boundary"),
        ]
    )
    second = InvariantSet(rules=list(reversed(first.rules)))

    assert first.hash == second.hash
    with pytest.raises(ValidationError, match="ID правила повторяется"):
        InvariantSet(rules=[first.rules[0], first.rules[0]])


def test_session_versions_rules_and_rejects_a_stale_editor(client):
    session_id = client.post("/api/sessions", json={}).json()["id"]
    initial = client.get(f"/api/sessions/{session_id}").json()

    assert initial["invariants"]["rules"] == []
    assert initial["invariant_history"][0]["origin"] == "create"

    updated = client.put(
        f"/api/sessions/{session_id}/invariants",
        json={
            "expected_version_hash": initial["invariants"]["version_hash"],
            "rules": rules(),
        },
    )
    assert updated.status_code == 200
    detail = updated.json()
    assert detail["invariants"]["rules"] == rules()
    assert detail["invariants"]["version_hash"] != initial["invariants"]["version_hash"]
    assert [item["origin"] for item in detail["invariant_history"]] == ["create", "edit"]

    stale = client.put(
        f"/api/sessions/{session_id}/invariants",
        json={
            "expected_version_hash": initial["invariants"]["version_hash"],
            "rules": rules("Python 3.10+"),
        },
    )
    assert stale.status_code == 409
    assert "уже изменён" in stale.json()["detail"]
    assert client.get(f"/api/sessions/{session_id}").json()["invariants"] == detail["invariants"]


def test_active_task_requires_explicit_replanning(client):
    session_id = client.post("/api/sessions", json={}).json()["id"]
    task = repo.create_task(client.app.state.db, session_id)
    repo.save_task_plan(client.app.state.db, task["id"], "1. Реализовать\n\nКритерии:\n- тесты")
    repo.apply_transition(client.app.state.db, task["id"], "execution")
    initial = client.get(f"/api/sessions/{session_id}").json()
    payload = {
        "expected_version_hash": initial["invariants"]["version_hash"],
        "rules": rules(),
    }

    refused = client.put(f"/api/sessions/{session_id}/invariants", json=payload)
    assert refused.status_code == 409
    assert "возврата в планирование" in refused.json()["detail"]
    assert client.get(f"/api/sessions/{session_id}").json()["task"]["stage"] == "execution"

    applied = client.put(
        f"/api/sessions/{session_id}/invariants",
        json={**payload, "replan_task": True},
    )
    assert applied.status_code == 200
    task = applied.json()["task"]
    assert task["stage"] == "planning"
    assert task["steps"] == [] and task["criteria"] == [] and task["verdict"] is None
    assert task["transitions"][-1]["kind"] == "invariants"


def test_restart_and_fork_keep_independent_version_pointers(tmp_path):
    path = tmp_path / "invariants.db"
    db = init_db(path)
    session_id = repo.create_session(db, AgentSpec())
    initial = repo.session_invariants(db, session_id)
    first = InvariantSet(rules=[InvariantRule(**rules()[0])])
    assert repo.set_session_invariants(
        db, session_id, first, expected_hash=initial["version_hash"]
    ) == "updated"
    repo.append_message(db, session_id, "user", "вопрос")
    repo.append_message(db, session_id, "assistant", "ответ")
    branch_id = repo.fork_session(db, session_id, 2)
    db.close()

    reopened = init_db(path)
    assert repo.session_invariants(reopened, session_id)["version_hash"] == first.hash
    assert repo.session_invariants(reopened, branch_id)["version_hash"] == first.hash
    second = InvariantSet(rules=[InvariantRule(**rules("Python 3.10+")[0])])
    assert repo.set_session_invariants(
        reopened, session_id, second, expected_hash=first.hash
    ) == "updated"
    assert repo.session_invariants(reopened, branch_id)["version_hash"] == first.hash
    assert repo.invariant_history(reopened, branch_id)[0]["origin"] == "fork"
    assert reopened.execute("PRAGMA foreign_key_check").fetchall() == []


def test_fork_drops_an_approval_for_a_different_invariant_version(tmp_path):
    db = init_db(tmp_path / "fork-rules.db")
    session_id = repo.create_session(db, AgentSpec())
    first = InvariantSet(rules=[InvariantRule(**rules()[0])])
    initial = repo.session_invariants(db, session_id)
    assert repo.set_session_invariants(
        db, session_id, first, expected_hash=initial["version_hash"]
    ) == "updated"
    repo.append_message(db, session_id, "user", "задача")
    repo.append_message(db, session_id, "assistant", "принято")
    task = repo.create_task(db, session_id, seq=2)
    repo.save_task_plan(db, task["id"], "1. Сделать", seq=2)
    approved = repo.apply_transition(db, task["id"], "execution", seq=2)
    assert approved["approved_invariant_hash"] == first.hash
    repo.append_message(db, session_id, "user", "новые правила")
    second = InvariantSet(rules=[InvariantRule(**rules("Python 3.10+")[0])])
    assert repo.set_session_invariants(
        db,
        session_id,
        second,
        expected_hash=first.hash,
        replan_task=True,
    ) == "updated"

    branch_id = repo.fork_session(db, session_id, 2)
    branch = repo.active_task(db, branch_id)

    assert repo.session_invariants(db, branch_id)["version_hash"] == second.hash
    assert branch["stage"] == "planning"
    assert branch["approved_plan_hash"] is None
    assert branch["approved_invariant_hash"] is None


def test_run_and_ledger_keep_the_fixed_version(tmp_path):
    db = init_db(tmp_path / "matrix.db")
    invariants = InvariantSet(rules=[InvariantRule(**rules()[0])])
    spec = AgentSpec()
    run_id = repo.create_run(
        db,
        name="rules",
        prompt="question",
        repeats=1,
        budget_usd=None,
        estimated_cost_usd=0,
        specs=[spec],
        invariant_sets=[invariants],
    )
    repo.record_run_usage(
        db,
        run_id,
        spec,
        spec.model.model,
        Usage(tokens_in=1, tokens_out=1),
        latency_ms=1,
        invariant_hash=invariants.hash,
    )

    report = repo.get_run(db, run_id)
    ledger = db.execute(
        "SELECT invariant_hash FROM ledger WHERE run_id = ?", (run_id,)
    ).fetchone()
    assert report["configs"][0]["invariants"] == {
        "hash": invariants.hash,
        "rules": rules(),
    }
    assert ledger["invariant_hash"] == invariants.hash


def test_v16_upgrade_assigns_the_empty_set(tmp_path):
    path = tmp_path / "v16.db"
    db = init_db(path)
    session_id = repo.create_session(db, AgentSpec())
    db.execute("DELETE FROM session_invariant_assignments")
    db.execute("UPDATE sessions SET invariant_hash = NULL")
    db.execute("PRAGMA user_version = 16")
    db.commit()
    db.close()

    upgraded = init_db(path)
    stored = repo.session_invariants(upgraded, session_id)
    assert stored == {"version_hash": InvariantSet().hash, "rules": []}
    assert repo.invariant_history(upgraded, session_id)[0]["origin"] == "migration"
    assert upgraded.execute("PRAGMA foreign_key_check").fetchall() == []


def test_v19_upgrade_does_not_invent_task_approvals(tmp_path):
    path = tmp_path / "v18.db"
    db = init_db(path)
    session_id = repo.create_session(db, AgentSpec())
    task = repo.create_task(db, session_id)
    repo.save_task_plan(db, task["id"], "1. Существующий план")
    for column in (
        "verdict_result_revision",
        "approved_invariant_hash",
        "approved_plan_hash",
        "result_revision",
        "revision",
    ):
        db.execute(f"ALTER TABLE tasks DROP COLUMN {column}")
    db.execute("PRAGMA user_version = 18")
    db.commit()
    db.close()

    upgraded = init_db(path)
    restored = repo.active_task(upgraded, session_id)

    assert restored["revision"] == 0
    assert restored["result_revision"] == 0
    assert restored["approved_plan_hash"] is None
    assert restored["approved_invariant_hash"] is None
    assert restored["verdict_result_revision"] is None
    assert upgraded.execute("PRAGMA foreign_key_check").fetchall() == []
