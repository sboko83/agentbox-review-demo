import json
import sqlite3

from fastapi.testclient import TestClient

from agentbox.api.app import create_app
from agentbox.core.context import LAYER_KINDS
from agentbox.core.spec import AgentSpec
from agentbox.store import repository
from agentbox.store.db import SCHEMA_VERSION, init_db


def test_health_reports_schema_and_key(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTBOX_DB", str(tmp_path / "test.db"))
    with TestClient(create_app()) as client:
        body = client.get("/api/health").json()

    assert body["status"] in {"ok", "degraded"}
    assert body["version"]
    assert set(body["providers"]) == {"zen"}
    assert set(body["tables"]) == {
        "agent_specs",
        "checkpoints",
        "agent_presets",
        "runs",
        "sessions",
        "session_summaries",
        "memory_proposals",
        "memory_scopes",
        "memory_versions",
        "messages",
        "tool_calls",
        "ledger",
        "user_profile_versions",
        "user_profiles",
        "run_configurations",
        "run_results",
        "tasks",
        "task_transitions",
        "invariant_versions",
        "session_invariant_assignments",
        "invariant_checks",
    }


def test_v7_database_rehashes_specs_and_moves_session_tools(tmp_path):
    path = tmp_path / "v7.db"
    old = sqlite3.connect(path)
    old.executescript(
        """
        PRAGMA user_version = 7;
        CREATE TABLE agent_specs (
            hash TEXT PRIMARY KEY, name TEXT NOT NULL, spec_json TEXT NOT NULL, created_at TEXT NOT NULL
        );
        CREATE TABLE runs (
            id TEXT PRIMARY KEY, name TEXT, prompt TEXT NOT NULL, repeats INTEGER NOT NULL,
            budget_usd REAL, estimated_cost_usd REAL, status TEXT NOT NULL,
            created_at TEXT NOT NULL, finished_at TEXT
        );
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY, spec_hash TEXT NOT NULL REFERENCES agent_specs(hash),
            run_id TEXT REFERENCES runs(id), title TEXT, workspace_path TEXT,
            executor_kind TEXT NOT NULL DEFAULT 'local', planning_mode TEXT NOT NULL DEFAULT 'execution',
            plan TEXT, chars_per_token REAL NOT NULL DEFAULT 3.0, status TEXT NOT NULL,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL REFERENCES sessions(id),
            seq INTEGER NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL, created_at TEXT NOT NULL
        );
        CREATE TABLE ledger (
            id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT REFERENCES sessions(id),
            run_id TEXT REFERENCES runs(id), model TEXT NOT NULL, input_tokens INTEGER NOT NULL,
            output_tokens INTEGER NOT NULL, reasoning_tokens INTEGER NOT NULL, cost_usd REAL NOT NULL,
            latency_ms INTEGER, created_at TEXT NOT NULL
        );
        CREATE TABLE profiles (
            id TEXT PRIMARY KEY, name TEXT NOT NULL UNIQUE,
            spec_hash TEXT NOT NULL REFERENCES agent_specs(hash), is_default INTEGER NOT NULL,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        CREATE TABLE run_configurations (
            run_id TEXT NOT NULL REFERENCES runs(id), config_index INTEGER NOT NULL,
            spec_hash TEXT NOT NULL REFERENCES agent_specs(hash), PRIMARY KEY (run_id, config_index)
        );
        """
    )
    legacy = {
        "name": "legacy",
        "model": "gpt-5-nano",
        "role": "Старая роль",
        "context_strategy": "summary",
        "context_overflow": "trim",
        "keep_last_messages": 3,
    }
    old.execute(
        "INSERT INTO agent_specs VALUES ('old', 'legacy', ?, '2026-01-01')",
        (json.dumps(legacy),),
    )
    old.execute(
        "INSERT INTO sessions VALUES "
        "('session', 'old', NULL, 'old', '/tmp', 'docker', 'execution', NULL, 3.0, 'active', '2026-01-01', '2026-01-01')"
    )
    old.execute(
        "INSERT INTO profiles VALUES ('profile', 'legacy', 'old', 1, '2026-01-01', '2026-01-01')"
    )
    old.execute(
        "INSERT INTO runs VALUES "
        "('run', 'legacy', 'задача', 1, NULL, 0.1, 'completed', '2026-01-01', '2026-01-01')"
    )
    old.execute("INSERT INTO run_configurations VALUES ('run', 0, 'old')")
    old.execute(
        "INSERT INTO messages VALUES (1, 'session', 1, 'user', 'вопрос', '2026-01-01')"
    )
    old.execute(
        "INSERT INTO messages VALUES (2, 'session', 2, 'assistant', 'ответ', '2026-01-01')"
    )
    old.execute(
        "INSERT INTO ledger VALUES (1, 'session', NULL, 'gpt-5-nano', 10, 2, 0, 0.1, NULL, '2026-01-01')"
    )
    old.execute(
        "INSERT INTO ledger VALUES (2, NULL, 'run', 'gpt-5-nano', 10, 2, 0, 0.1, NULL, '2026-01-01')"
    )
    old.commit()
    old.close()

    migrated = init_db(path)
    session = migrated.execute("SELECT * FROM sessions WHERE id = 'session'").fetchone()
    raw = migrated.execute(
        "SELECT spec_json FROM agent_specs WHERE hash = ?", (session["spec_hash"],)
    ).fetchone()["spec_json"]
    spec = AgentSpec.model_validate_json(raw)
    usage = migrated.execute(
        "SELECT spec_hash, message_id FROM ledger WHERE session_id = 'session'"
    ).fetchone()
    preset_hash = migrated.execute(
        "SELECT spec_hash FROM agent_presets WHERE id = 'profile'"
    ).fetchone()["spec_hash"]
    config = migrated.execute(
        "SELECT name, spec_hash FROM run_configurations WHERE run_id = 'run'"
    ).fetchone()
    run_usage_hash = migrated.execute(
        "SELECT spec_hash FROM ledger WHERE run_id = 'run'"
    ).fetchone()["spec_hash"]

    assert migrated.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    assert "session_summaries" in {
        row[0]
        for row in migrated.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }
    assert "memory_versions" in {
        row[0]
        for row in migrated.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }
    assert {"parent_id", "fork_seq"} <= set(session.keys())
    assert "workspace_path" not in session.keys()
    assert "executor_kind" not in session.keys()
    assert "name" not in {
        row["name"]
        for row in migrated.execute("PRAGMA table_info(agent_specs)").fetchall()
    }
    assert spec.model.role == "Старая роль"
    assert spec.context.mode == "summary"
    assert spec.context.overflow == "drop"
    assert spec.tools.workspace_path == "/tmp"
    assert spec.tools.executor == "docker"
    assert usage["spec_hash"] == session["spec_hash"]
    assert usage["message_id"] == 2
    assert preset_hash == config["spec_hash"] != "old"
    assert config["name"] == "legacy"
    assert run_usage_hash == config["spec_hash"]
    assert migrated.execute("PRAGMA foreign_key_check").fetchall() == []
    migrated.close()


def test_v10_database_folds_facts_into_the_working_memory_layer(tmp_path):
    path = tmp_path / "v10.db"
    old = sqlite3.connect(path)
    old.executescript(
        """
        PRAGMA user_version = 10;
        CREATE TABLE agent_specs (
            hash TEXT PRIMARY KEY, spec_json TEXT NOT NULL, created_at TEXT NOT NULL
        );
        CREATE TABLE profiles (
            id TEXT PRIMARY KEY, name TEXT NOT NULL UNIQUE,
            spec_hash TEXT NOT NULL REFERENCES agent_specs(hash),
            is_default INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        CREATE TABLE runs (
            id TEXT PRIMARY KEY, name TEXT, prompt TEXT NOT NULL, repeats INTEGER NOT NULL,
            budget_usd REAL, estimated_cost_usd REAL, status TEXT NOT NULL,
            created_at TEXT NOT NULL, finished_at TEXT
        );
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY, spec_hash TEXT NOT NULL REFERENCES agent_specs(hash),
            run_id TEXT REFERENCES runs(id), parent_id TEXT, fork_seq INTEGER, title TEXT,
            planning_mode TEXT NOT NULL DEFAULT 'execution', plan TEXT,
            chars_per_token REAL NOT NULL DEFAULT 3.0, status TEXT NOT NULL,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL REFERENCES sessions(id),
            seq INTEGER NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL, created_at TEXT NOT NULL
        );
        CREATE TABLE session_facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL REFERENCES sessions(id),
            version INTEGER NOT NULL, covered_seq INTEGER NOT NULL,
            facts_json TEXT NOT NULL, tokens INTEGER NOT NULL, created_at TEXT NOT NULL
        );
        CREATE TABLE ledger (
            id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT REFERENCES sessions(id),
            run_id TEXT, model TEXT NOT NULL, kind TEXT NOT NULL DEFAULT 'chat',
            tokens_in INTEGER NOT NULL DEFAULT 0, tokens_out INTEGER NOT NULL DEFAULT 0,
            reasoning INTEGER NOT NULL DEFAULT 0, cost_usd REAL NOT NULL DEFAULT 0,
            spec_hash TEXT REFERENCES agent_specs(hash), message_id INTEGER,
            created_at TEXT NOT NULL
        );
        CREATE TABLE run_configurations (
            run_id TEXT NOT NULL, config_index INTEGER NOT NULL, name TEXT,
            spec_hash TEXT NOT NULL REFERENCES agent_specs(hash)
        );
        """
    )
    legacy = {
        "model": {"provider": "zen", "model": "gpt-5-nano", "role": "Старая роль"},
        "context": {"mode": "facts", "keep_last_messages": 4, "facts_model": "gpt-5-nano"},
    }
    old.execute(
        "INSERT INTO agent_specs (hash, spec_json, created_at) VALUES ('old', ?, 'now')",
        (json.dumps(legacy, ensure_ascii=False),),
    )
    old.execute(
        "INSERT INTO sessions (id, spec_hash, title, status, created_at, updated_at)"
        " VALUES ('s1', 'old', 'диалог', 'active', 'now', 'now')"
    )
    old.execute(
        "INSERT INTO profiles (id, name, spec_hash, created_at, updated_at)"
        " VALUES ('p1', 'старый', 'old', 'now', 'now')"
    )
    old.execute(
        "INSERT INTO session_facts"
        " (session_id, version, covered_seq, facts_json, tokens, created_at)"
        " VALUES ('s1', 1, 2, ?, 3, 'now')",
        (json.dumps({"name": "Сергей"}, ensure_ascii=False),),
    )
    old.execute(
        "INSERT INTO ledger (session_id, model, kind, spec_hash, created_at)"
        " VALUES ('s1', 'gpt-5-nano', 'facts', 'old', 'now')"
    )
    old.commit()
    old.close()

    migrated = init_db(path)
    session = migrated.execute("SELECT * FROM sessions WHERE id = 's1'").fetchone()
    spec = repository.get_spec(migrated, session["spec_hash"])
    version = migrated.execute("SELECT * FROM memory_versions").fetchone()

    assert migrated.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    assert "session_facts" not in {
        row[0]
        for row in migrated.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }
    # The retired mode becomes a window plus a layer, and the payload it
    # produces is the same one it produced before.
    assert spec is not None
    assert spec.context.mode == "window" and spec.context.keep_last_messages == 4
    assert spec.memory.working_enabled is True
    assert session["spec_hash"] == spec.hash != "old"
    assert migrated.execute(
        "SELECT spec_hash FROM agent_presets WHERE id = 'p1'"
    ).fetchone()["spec_hash"] == spec.hash
    assert version["layer"] == "working" and version["origin"] == "migration"
    migrated_item = json.loads(version["items_json"])["name"]
    assert migrated_item["value"] == "Сергей"
    # A kindless fact has to land under a kind of its layer, or every later
    # merge and hand write would refuse the item it migrated.
    assert migrated_item["kind"] in LAYER_KINDS["working"]
    assert migrated.execute("SELECT kind FROM ledger").fetchone()["kind"] == "memory"
    assert migrated.execute(
        "SELECT name FROM memory_scopes WHERE id = ?", (session["memory_scope_id"],)
    ).fetchone()["name"] == "default"
    assert migrated.execute("PRAGMA foreign_key_check").fetchall() == []


def _v12_database(path):
    """Take the current schema back to v12: presets were `profiles`, no users."""
    conn = init_db(path)
    conn.execute("DROP INDEX idx_memory_proposals_profile")
    conn.execute(
        """
        CREATE TABLE proposals_v12 (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            layer TEXT NOT NULL CHECK (layer IN ('working', 'longterm')),
            session_id TEXT REFERENCES sessions(id) ON DELETE CASCADE,
            scope_id TEXT REFERENCES memory_scopes(id) ON DELETE CASCADE,
            key TEXT NOT NULL, value TEXT NOT NULL, kind TEXT NOT NULL,
            source_seq INTEGER NOT NULL, created_at TEXT NOT NULL,
            CHECK ((session_id IS NULL) <> (scope_id IS NULL))
        )
        """
    )
    conn.execute("DROP TABLE memory_proposals")
    conn.execute("ALTER TABLE proposals_v12 RENAME TO memory_proposals")
    conn.execute(
        "CREATE UNIQUE INDEX idx_memory_proposals_session"
        " ON memory_proposals (session_id, layer, key) WHERE session_id IS NOT NULL"
    )
    conn.execute("ALTER TABLE sessions DROP COLUMN user_profile_id")
    conn.execute("ALTER TABLE ledger DROP COLUMN user_profile_hash")
    conn.execute("ALTER TABLE run_configurations DROP COLUMN user_profile_hash")
    conn.execute("DROP TABLE user_profiles")
    conn.execute("DROP TABLE user_profile_versions")
    conn.execute("DROP INDEX idx_agent_presets_default")
    conn.execute("ALTER TABLE agent_presets RENAME TO profiles")
    conn.execute(
        "CREATE UNIQUE INDEX idx_profiles_default ON profiles (is_default)"
        " WHERE is_default = 1"
    )
    conn.execute("PRAGMA user_version = 12")
    conn.commit()
    return conn


def test_v12_database_renames_profiles_to_presets(tmp_path):
    path = tmp_path / "v12.db"
    old = _v12_database(path)
    digest = repository.save_spec(old, AgentSpec())
    old.execute(
        "INSERT INTO profiles VALUES ('p1', 'основной', ?, 1, '2026-01-01', '2026-01-01')",
        (digest,),
    )
    old.execute(
        "INSERT INTO sessions (id, spec_hash, status, created_at, updated_at)"
        " VALUES ('s1', ?, 'active', '2026-01-01', '2026-01-01')",
        (digest,),
    )
    old.execute(
        "INSERT INTO memory_proposals"
        " (layer, session_id, key, value, kind, source_seq, created_at)"
        " VALUES ('working', 's1', 'goal', 'отчёт', 'goal', 2, '2026-01-01')"
    )
    old.commit()
    old.close()

    migrated = init_db(path)
    tables = {
        row[0]
        for row in migrated.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'index')"
        ).fetchall()
    }

    assert migrated.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    assert "profiles" not in tables and "idx_profiles_default" not in tables
    assert {"agent_presets", "idx_agent_presets_default"} <= tables
    assert {"user_profiles", "user_profile_versions", "idx_memory_proposals_profile"} <= tables
    assert migrated.execute("SELECT user_profile_id FROM sessions").fetchone()[0] is None
    for table in ("ledger", "run_configurations"):
        columns = {row[1] for row in migrated.execute(f"PRAGMA table_info({table})")}
        assert "user_profile_hash" in columns
    # Pending proposals survive the rebuild that let a profile own one.
    assert [
        tuple(row)
        for row in migrated.execute("SELECT session_id, key, value FROM memory_proposals")
    ] == [("s1", "goal", "отчёт")]
    preset = repository.get_default_preset(migrated)
    assert preset is not None and preset["id"] == "p1" and preset["name"] == "основной"
    # The default stays unique after the rename: the index was rebuilt.
    try:
        migrated.execute(
            "INSERT INTO agent_presets VALUES ('p2', 'второй', ?, 1, '2026-01-01', '2026-01-01')",
            (digest,),
        )
    except sqlite3.IntegrityError:
        pass
    else:
        raise AssertionError("second default preset was accepted")
    assert migrated.execute("PRAGMA foreign_key_check").fetchall() == []
    migrated.close()
