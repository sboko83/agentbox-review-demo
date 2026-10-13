"""SQLite access: connection setup and schema application.

One file, no server, no ORM. Runs cost money and must survive a restart, so
persistence exists from the start rather than being bolted on later.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from agentbox.core.spec import AgentSpec
from agentbox.core.invariants import InvariantSet
from agentbox.core.task_machine import TaskState, plan_state

SCHEMA_FILE = Path(__file__).resolve().parent / "schema.sql"
SCHEMA_VERSION = 19


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def init_db(db_path: Path) -> sqlite3.Connection:
    """Apply the schema once; `user_version` marks it as done."""
    conn = connect(db_path)
    current = conn.execute("PRAGMA user_version").fetchone()[0]
    if current < SCHEMA_VERSION:
        if 0 < current < 13:
            # Before the schema script: it would otherwise create an empty
            # `agent_presets` next to the old table, and index proposal
            # columns the old table does not have.
            _rename_presets(conn)
            _rebuild_proposals(conn)
        conn.executescript(SCHEMA_FILE.read_text(encoding="utf-8"))
        if current == 0:
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ledger_message ON ledger (message_id)")
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            conn.commit()
            return conn
        # SQLite's CREATE TABLE IF NOT EXISTS does not add columns to databases
        # created by an earlier phase, so keep this small migration explicit.
        session_columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(sessions)").fetchall()
        }
        if "executor_kind" not in session_columns:
            conn.execute(
                "ALTER TABLE sessions ADD COLUMN executor_kind TEXT NOT NULL DEFAULT 'local'"
            )
        if "chars_per_token" not in session_columns:
            conn.execute(
                "ALTER TABLE sessions ADD COLUMN chars_per_token REAL NOT NULL DEFAULT 3.0"
            )
        if "parent_id" not in session_columns:
            conn.execute(
                "ALTER TABLE sessions ADD COLUMN parent_id TEXT"
                " REFERENCES sessions(id) ON DELETE SET NULL"
            )
        if "fork_seq" not in session_columns:
            conn.execute("ALTER TABLE sessions ADD COLUMN fork_seq INTEGER")
        if "memory_covered_seq" not in session_columns:
            conn.execute(
                "ALTER TABLE sessions ADD COLUMN memory_covered_seq INTEGER NOT NULL"
                " DEFAULT 0"
            )
        if "memory_scope_id" not in session_columns:
            conn.execute(
                "ALTER TABLE sessions ADD COLUMN memory_scope_id TEXT"
                " REFERENCES memory_scopes(id) ON DELETE SET NULL"
            )
        if "user_profile_id" not in session_columns:
            conn.execute(
                "ALTER TABLE sessions ADD COLUMN user_profile_id TEXT"
                " REFERENCES user_profiles(id) ON DELETE SET NULL"
            )
        if "invariant_hash" not in session_columns:
            conn.execute(
                "ALTER TABLE sessions ADD COLUMN invariant_hash TEXT"
                " REFERENCES invariant_versions(hash)"
            )
        empty_invariants = InvariantSet()
        conn.execute(
            "INSERT OR IGNORE INTO invariant_versions (hash, body_json, created_at)"
            " VALUES (?, ?, ?)",
            (
                empty_invariants.hash,
                empty_invariants.canonical_json(),
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
            ),
        )
        conn.execute(
            "UPDATE sessions SET invariant_hash = ? WHERE invariant_hash IS NULL",
            (empty_invariants.hash,),
        )
        conn.execute(
            "INSERT INTO session_invariant_assignments"
            " (session_id, version_hash, previous_hash, origin, created_at)"
            " SELECT s.id, s.invariant_hash, NULL, 'migration', ? FROM sessions s"
            " WHERE NOT EXISTS (SELECT 1 FROM session_invariant_assignments a"
            "                   WHERE a.session_id = s.id)",
            (datetime.now(timezone.utc).isoformat(timespec="seconds"),),
        )
        ledger_columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(ledger)").fetchall()
        }
        ledger_additions = {
            "kind": "TEXT NOT NULL DEFAULT 'chat'",
            "role_estimate_tokens": "INTEGER NOT NULL DEFAULT 0",
            "history_estimate_tokens": "INTEGER NOT NULL DEFAULT 0",
            "request_estimate_tokens": "INTEGER NOT NULL DEFAULT 0",
            "input_estimate_tokens": "INTEGER NOT NULL DEFAULT 0",
            "context_estimate_tokens": "INTEGER NOT NULL DEFAULT 0",
            "context_input_tokens": "INTEGER NOT NULL DEFAULT 0",
            "context_output_tokens": "INTEGER NOT NULL DEFAULT 0",
            "context_limit_tokens": "INTEGER",
            "context_limit_source": "TEXT",
            "dropped_messages": "INTEGER NOT NULL DEFAULT 0",
            "spec_hash": "TEXT REFERENCES agent_specs(hash)",
            "message_id": "INTEGER REFERENCES messages(id) ON DELETE SET NULL",
            "user_profile_hash": "TEXT REFERENCES user_profile_versions(hash)",
            "invariant_hash": "TEXT REFERENCES invariant_versions(hash)",
        }
        for name, declaration in ledger_additions.items():
            if name not in ledger_columns:
                conn.execute(f"ALTER TABLE ledger ADD COLUMN {name} {declaration}")
        run_config_columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(run_configurations)").fetchall()
        }
        if "user_profile_hash" not in run_config_columns:
            conn.execute(
                "ALTER TABLE run_configurations ADD COLUMN user_profile_hash TEXT"
                " REFERENCES user_profile_versions(hash)"
            )
        if "invariant_hash" not in run_config_columns:
            conn.execute(
                "ALTER TABLE run_configurations ADD COLUMN invariant_hash TEXT"
                " REFERENCES invariant_versions(hash)"
            )
        if "name" not in run_config_columns:
            conn.execute("ALTER TABLE run_configurations ADD COLUMN name TEXT")
            conn.execute(
                "UPDATE run_configurations SET name = COALESCE(("
                " SELECT name FROM agent_specs WHERE hash = run_configurations.spec_hash"
                "), 'Конфигурация ' || (config_index + 1))"
            )
        if current < 8:
            try:
                _migrate_v8(conn)
            except Exception:
                conn.rollback()
                raise
        if current < 11 and "session_facts" in table_names(conn):
            try:
                _migrate_v11(conn)
            except Exception:
                conn.rollback()
                raise
        if current < 12:
            # Everything routed so far was routed from the working layer, so
            # its coverage is the watermark an upgraded session continues from.
            # It runs after the v11 fold, which is what creates those versions.
            conn.execute(
                "UPDATE sessions SET memory_covered_seq = COALESCE(("
                " SELECT MAX(covered_seq) FROM memory_versions"
                " WHERE memory_versions.session_id = sessions.id"
                "), 0)"
            )
        if current < 15:
            try:
                _migrate_v15(conn)
            except Exception:
                conn.rollback()
                raise
        task_columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(tasks)").fetchall()
        }
        if "covered_seq" not in task_columns:
            # v16: a task created before the tracker has read nothing, which is
            # exactly what the default says.
            conn.execute(
                "ALTER TABLE tasks ADD COLUMN covered_seq INTEGER NOT NULL DEFAULT 0"
            )
        task_additions = {
            "revision": "INTEGER NOT NULL DEFAULT 0",
            "result_revision": "INTEGER NOT NULL DEFAULT 0",
            "approved_plan_hash": "TEXT",
            "approved_invariant_hash": "TEXT REFERENCES invariant_versions(hash)",
            "verdict_result_revision": "INTEGER",
        }
        for name, declaration in task_additions.items():
            if name not in task_columns:
                conn.execute(f"ALTER TABLE tasks ADD COLUMN {name} {declaration}")
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        conn.commit()
    elif current in {8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19}:
        # A development reload may have opened the database while this
        # migration itself was still being implemented. Finish the final v8
        # shape idempotently instead of requiring a fake version bump.
        spec_columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(agent_specs)").fetchall()
        }
        if "name" in spec_columns:
            conn.execute("ALTER TABLE agent_specs DROP COLUMN name")
            conn.commit()
    return conn


def _migrate_v15(conn: sqlite3.Connection) -> None:
    """Planning mode becomes a task: the stage of a session has one owner.

    A session left in `planning` keeps its draft: it reopens as a task on the
    planning stage with that text, parsed into steps when it parses at all.
    """
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(sessions)").fetchall()}
    if "planning_mode" not in columns:
        return
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    planning = conn.execute(
        "SELECT id, plan FROM sessions WHERE planning_mode = 'planning'"
    ).fetchall()
    for session in planning:
        state = TaskState()
        if (session["plan"] or "").strip():
            state = plan_state(state, session["plan"])
        task_id = uuid4().hex
        seq = conn.execute(
            "SELECT COALESCE(MAX(seq), 0) AS seq FROM messages WHERE session_id = ?",
            (session["id"],),
        ).fetchone()["seq"]
        conn.execute(
            "INSERT INTO tasks (id, session_id, stage, paused, current_step, verdict,"
            " steps_json, criteria_json, plan_text, plan_error, created_at, updated_at)"
            " VALUES (?, ?, 'planning', 0, NULL, NULL, ?, ?, ?, ?, ?, ?)",
            (
                task_id,
                session["id"],
                json.dumps([item.to_dict() for item in state.steps], ensure_ascii=False),
                json.dumps(list(state.criteria), ensure_ascii=False),
                state.plan_text or None,
                state.plan_error,
                now,
                now,
            ),
        )
        conn.execute(
            "INSERT INTO task_transitions (task_id, kind, from_stage, to_stage, paused,"
            " actor, reason, seq, state_json, created_at)"
            " VALUES (?, 'migration', NULL, 'planning', 0, 'user', ?, ?, ?, ?)",
            (
                task_id,
                "перенос режима планирования",
                seq,
                json.dumps(state.to_dict(), ensure_ascii=False),
                now,
            ),
        )
    _rebuild_sessions(conn)


def _rebuild_sessions(conn: sqlite3.Connection) -> None:
    """Drop `planning_mode` and `plan` the only way SQLite allows: a copy.

    The new table is filled under a temporary name and renamed only after the
    old one is gone, so the foreign keys of `messages` and friends keep
    pointing at the name `sessions` throughout.
    """
    conn.commit()
    conn.execute("PRAGMA foreign_keys = OFF")
    try:
        conn.execute(
            """
            CREATE TABLE sessions_v15 (
                id             TEXT PRIMARY KEY,
                spec_hash      TEXT NOT NULL REFERENCES agent_specs(hash),
                run_id         TEXT REFERENCES runs(id) ON DELETE CASCADE,
                parent_id      TEXT REFERENCES sessions(id) ON DELETE SET NULL,
                fork_seq       INTEGER,
                title          TEXT,
                chars_per_token REAL NOT NULL DEFAULT 3.0,
                memory_scope_id TEXT REFERENCES memory_scopes(id) ON DELETE SET NULL,
                memory_covered_seq INTEGER NOT NULL DEFAULT 0,
                user_profile_id TEXT REFERENCES user_profiles(id) ON DELETE SET NULL,
                invariant_hash TEXT REFERENCES invariant_versions(hash),
                status         TEXT NOT NULL,
                created_at     TEXT NOT NULL,
                updated_at     TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "INSERT INTO sessions_v15"
            " (id, spec_hash, run_id, parent_id, fork_seq, title, chars_per_token,"
            "  memory_scope_id, memory_covered_seq, user_profile_id, invariant_hash, status,"
            "  created_at, updated_at)"
            " SELECT id, spec_hash, run_id, parent_id, fork_seq, title, chars_per_token,"
            "        memory_scope_id, memory_covered_seq, user_profile_id, invariant_hash, status,"
            "        created_at, updated_at FROM sessions"
        )
        conn.execute("DROP TABLE sessions")
        conn.execute("ALTER TABLE sessions_v15 RENAME TO sessions")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_run ON sessions (run_id)")
        conn.commit()
    finally:
        conn.execute("PRAGMA foreign_keys = ON")


def _rename_presets(conn: sqlite3.Connection) -> None:
    """A named spec pointer is a preset: "profile" now belongs to the user."""
    tables = set(table_names(conn))
    if "profiles" in tables and "agent_presets" not in tables:
        conn.execute("DROP INDEX IF EXISTS idx_profiles_default")
        conn.execute("ALTER TABLE profiles RENAME TO agent_presets")


def _rebuild_proposals(conn: sqlite3.Connection) -> None:
    """Let a proposal belong to a user profile: a new owner and a new layer.

    SQLite cannot alter a CHECK constraint, so the table is rebuilt. It runs
    before the schema script creates `user_profiles`, so foreign keys are off
    for the copy and the new table is created under its final name: both a
    checked insert and a rename would try to resolve the missing parent.
    """
    if "memory_proposals" not in table_names(conn):
        return
    columns = {
        row["name"]
        for row in conn.execute("PRAGMA table_info(memory_proposals)").fetchall()
    }
    if "user_profile_id" in columns:
        return
    conn.commit()
    conn.execute("PRAGMA foreign_keys = OFF")
    try:
        _copy_proposals(conn)
        conn.commit()
    finally:
        conn.execute("PRAGMA foreign_keys = ON")


def _copy_proposals(conn: sqlite3.Connection) -> None:
    conn.execute("ALTER TABLE memory_proposals RENAME TO memory_proposals_v12")
    conn.execute(
        """
        CREATE TABLE memory_proposals (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            layer      TEXT NOT NULL CHECK (layer IN ('working', 'longterm', 'profile')),
            session_id TEXT REFERENCES sessions(id) ON DELETE CASCADE,
            scope_id   TEXT REFERENCES memory_scopes(id) ON DELETE CASCADE,
            user_profile_id TEXT REFERENCES user_profiles(id) ON DELETE CASCADE,
            key        TEXT NOT NULL,
            value      TEXT NOT NULL,
            kind       TEXT NOT NULL,
            source_seq INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            CHECK ((session_id IS NOT NULL) + (scope_id IS NOT NULL)
                   + (user_profile_id IS NOT NULL) = 1)
        )
        """
    )
    conn.execute(
        "INSERT INTO memory_proposals"
        " (id, layer, session_id, scope_id, key, value, kind, source_seq, created_at)"
        " SELECT id, layer, session_id, scope_id, key, value, kind, source_seq,"
        " created_at FROM memory_proposals_v12"
    )
    conn.execute("DROP TABLE memory_proposals_v12")


def _migrate_v11(conn: sqlite3.Connection) -> None:
    """Fold facts into the working memory layer and re-address every spec.

    `AgentSpec` gained a `memory` section, so the canonical JSON of every
    stored spec changed and so did its hash. Leaving the old digests in place
    would fork each configuration in two the first time it is edited.
    """
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    conn.execute(
        "INSERT OR IGNORE INTO memory_scopes (id, name, created_at)"
        " VALUES (?, 'default', ?)",
        (uuid4().hex, now),
    )
    scope_id = conn.execute(
        "SELECT id FROM memory_scopes WHERE name = 'default'"
    ).fetchone()["id"]
    conn.execute(
        "UPDATE sessions SET memory_scope_id = ? WHERE memory_scope_id IS NULL",
        (scope_id,),
    )

    facts = conn.execute(
        "SELECT session_id, version, covered_seq, facts_json, tokens, created_at"
        " FROM session_facts ORDER BY session_id, version"
    ).fetchall()
    for row in facts:
        # The old memory had no vocabulary of kinds, but every later check has
        # one: an item under a kind outside the layer is refused by the merge
        # and by every hand write. `artifact` is the working kind for data the
        # task produced, which is what these key-value facts were; the origin
        # records that nobody actually routed them.
        items = {
            key: {
                "value": str(value),
                "kind": "artifact",
                "origin": "migration",
                "seq": row["covered_seq"],
            }
            for key, value in json.loads(row["facts_json"]).items()
        }
        conn.execute(
            "INSERT INTO memory_versions"
            " (layer, session_id, version, covered_seq, origin, items_json,"
            "  tokens, created_at)"
            " VALUES ('working', ?, ?, ?, 'migration', ?, ?, ?)",
            (
                row["session_id"],
                row["version"],
                row["covered_seq"],
                json.dumps(items, ensure_ascii=False, sort_keys=True),
                row["tokens"],
                row["created_at"],
            ),
        )
    conn.execute("UPDATE ledger SET kind = 'memory' WHERE kind = 'facts'")

    migrated: dict[str, str] = {}
    for row in conn.execute(
        "SELECT hash, spec_json, created_at FROM agent_specs"
    ).fetchall():
        spec = AgentSpec.model_validate(json.loads(row["spec_json"]))
        if spec.hash == row["hash"]:
            continue
        conn.execute(
            "INSERT OR IGNORE INTO agent_specs (hash, spec_json, created_at)"
            " VALUES (?, ?, ?)",
            (spec.hash, spec.model_dump_json(), row["created_at"]),
        )
        migrated[row["hash"]] = spec.hash
    for table in ("sessions", "agent_presets", "run_configurations", "ledger"):
        for old_hash, digest in migrated.items():
            conn.execute(
                f"UPDATE {table} SET spec_hash = ? WHERE spec_hash = ?",
                (digest, old_hash),
            )
    for old_hash in migrated:
        conn.execute("DELETE FROM agent_specs WHERE hash = ?", (old_hash,))

    conn.execute("DROP TABLE IF EXISTS session_facts")


def _migrate_v8(conn: sqlite3.Connection) -> None:
    """Re-address flat specs and move per-session tool settings into them."""
    rows = conn.execute(
        "SELECT hash, name, spec_json, created_at FROM agent_specs"
    ).fetchall()
    old_specs = {row["hash"]: row for row in rows}
    migrated: dict[str, str] = {}

    def save(spec: AgentSpec, name: str, created_at: str) -> str:
        conn.execute(
            "INSERT OR IGNORE INTO agent_specs (hash, name, spec_json, created_at)"
            " VALUES (?, ?, ?, ?)",
            (spec.hash, name, spec.model_dump_json(), created_at),
        )
        return spec.hash

    for old_hash, row in old_specs.items():
        spec = AgentSpec.model_validate(json.loads(row["spec_json"]))
        migrated[old_hash] = save(spec, row["name"], row["created_at"])

    session_columns = {
        row["name"] for row in conn.execute("PRAGMA table_info(sessions)").fetchall()
    }
    has_legacy_tools = {"workspace_path", "executor_kind"} <= session_columns
    session_query = "SELECT id, spec_hash"
    if has_legacy_tools:
        session_query += ", workspace_path, executor_kind"
    session_query += " FROM sessions"
    for session in conn.execute(session_query).fetchall():
        old = old_specs.get(session["spec_hash"])
        if old is None:
            continue
        payload = json.loads(old["spec_json"])
        if has_legacy_tools:
            payload["workspace_path"] = session["workspace_path"]
            payload["executor_kind"] = session["executor_kind"]
        spec = AgentSpec.model_validate(payload)
        digest = save(spec, old["name"], old["created_at"])
        conn.execute("UPDATE sessions SET spec_hash = ? WHERE id = ?", (digest, session["id"]))

    for table in ("agent_presets", "run_configurations"):
        for old_hash, digest in migrated.items():
            conn.execute(
                f"UPDATE {table} SET spec_hash = ? WHERE spec_hash = ?", (digest, old_hash)
            )

    conn.execute(
        "UPDATE ledger SET spec_hash = ("
        " SELECT spec_hash FROM sessions WHERE sessions.id = ledger.session_id"
        ") WHERE spec_hash IS NULL AND session_id IS NOT NULL"
    )
    _backfill_run_ledger_specs(conn, save)
    _backfill_ledger_messages(conn)

    if has_legacy_tools:
        conn.execute("ALTER TABLE sessions DROP COLUMN workspace_path")
        conn.execute("ALTER TABLE sessions DROP COLUMN executor_kind")

    for old_hash in migrated:
        conn.execute("DELETE FROM agent_specs WHERE hash = ?", (old_hash,))
    conn.execute("ALTER TABLE agent_specs DROP COLUMN name")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ledger_message ON ledger (message_id)")


def _backfill_run_ledger_specs(conn: sqlite3.Connection, save) -> None:
    configs: dict[str, list[tuple[str, AgentSpec]]] = {}
    rows = conn.execute(
        "SELECT c.run_id, c.spec_hash, a.spec_json FROM run_configurations c"
        " JOIN agent_specs a ON a.hash = c.spec_hash ORDER BY c.config_index"
    ).fetchall()
    for row in rows:
        configs.setdefault(row["run_id"], []).append(
            (row["spec_hash"], AgentSpec.model_validate_json(row["spec_json"]))
        )
    ledger_rows = conn.execute(
        "SELECT id, run_id, model, created_at FROM ledger"
        " WHERE spec_hash IS NULL AND run_id IS NOT NULL ORDER BY id"
    ).fetchall()
    for row in ledger_rows:
        candidates = configs.get(row["run_id"], [])
        digest = next(
            (
                digest
                for digest, spec in candidates
                if row["model"]
                in {spec.model.model, spec.judge.model, spec.judge.fallback_model}
            ),
            candidates[0][0] if candidates else None,
        )
        if digest is None:
            spec = AgentSpec(model={"model": row["model"]})
            digest = save(spec, row["model"], row["created_at"])
        conn.execute("UPDATE ledger SET spec_hash = ? WHERE id = ?", (digest, row["id"]))


def _backfill_ledger_messages(conn: sqlite3.Connection) -> None:
    sessions = conn.execute(
        "SELECT DISTINCT session_id FROM ledger WHERE session_id IS NOT NULL"
    ).fetchall()
    for session in sessions:
        session_id = session["session_id"]
        ledger_ids = [
            row["id"]
            for row in conn.execute(
                "SELECT id FROM ledger WHERE session_id = ? AND kind = 'chat' ORDER BY id",
                (session_id,),
            ).fetchall()
        ]
        message_ids = [
            row["id"]
            for row in conn.execute(
                "SELECT id FROM messages WHERE session_id = ? AND role = 'assistant' ORDER BY seq",
                (session_id,),
            ).fetchall()
        ]
        for ledger_id, message_id in zip(ledger_ids, message_ids):
            conn.execute(
                "UPDATE ledger SET message_id = ? WHERE id = ?", (message_id, ledger_id)
            )


def table_names(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' "
        "AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ).fetchall()
    return [row["name"] for row in rows]
