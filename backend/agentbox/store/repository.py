"""Persistence for specs, sessions, messages and spend.

Plain SQL over sqlite3: the schema is small and the queries are few, so an ORM
would add a layer without removing one.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone

from agentbox.core.ports import ToolCall, ToolResult, Usage
from agentbox.core.invariants import InvariantSet
from agentbox.core.spec import AgentSpec
from agentbox.core.task_machine import (
    Step,
    TaskError,
    TaskState,
    plan_state,
    set_paused,
    set_step,
    set_verdict,
    transition,
)
from agentbox.core.user_profile import UserProfile


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _new_id() -> str:
    return uuid.uuid4().hex


def _load_spec(raw: str) -> AgentSpec:
    """Decode specs written before required fields received defaults."""
    data = json.loads(raw)
    if data.get("max_output_tokens") is None:
        data.pop("max_output_tokens", None)
    return AgentSpec(**data)


def _load_spec_dict(raw: str) -> dict:
    return _load_spec(raw).model_dump()


def _save_spec(conn: sqlite3.Connection, spec: AgentSpec) -> str:
    digest = spec.hash
    conn.execute(
        "INSERT OR IGNORE INTO agent_specs (hash, spec_json, created_at) VALUES (?, ?, ?)",
        (digest, spec.model_dump_json(), _now()),
    )
    return digest


def save_spec(conn: sqlite3.Connection, spec: AgentSpec) -> str:
    """Store a spec under its hash. Storing the same spec twice is a no-op."""
    digest = _save_spec(conn, spec)
    conn.commit()
    return digest


def get_spec(conn: sqlite3.Connection, digest: str) -> AgentSpec | None:
    row = conn.execute(
        "SELECT spec_json FROM agent_specs WHERE hash = ?", (digest,)
    ).fetchone()
    return _load_spec(row["spec_json"]) if row else None


# --- session invariants ---------------------------------------------------


def save_invariant_version(conn: sqlite3.Connection, invariants: InvariantSet) -> str:
    """Store one complete immutable set; equal content reuses the same row."""
    digest = invariants.hash
    conn.execute(
        "INSERT OR IGNORE INTO invariant_versions (hash, body_json, created_at)"
        " VALUES (?, ?, ?)",
        (digest, invariants.canonical_json(), _now()),
    )
    return digest


def get_invariant_version(
    conn: sqlite3.Connection, digest: str
) -> InvariantSet | None:
    row = conn.execute(
        "SELECT body_json FROM invariant_versions WHERE hash = ?", (digest,)
    ).fetchone()
    return InvariantSet.model_validate_json(row["body_json"]) if row else None


def session_invariants(conn: sqlite3.Connection, session_id: str) -> dict | None:
    row = conn.execute(
        "SELECT s.invariant_hash, v.body_json FROM sessions s"
        " LEFT JOIN invariant_versions v ON v.hash = s.invariant_hash"
        " WHERE s.id = ?",
        (session_id,),
    ).fetchone()
    if row is None:
        return None
    invariants = (
        InvariantSet.model_validate_json(row["body_json"])
        if row["body_json"]
        else InvariantSet()
    )
    return {"version_hash": row["invariant_hash"] or invariants.hash, **invariants.model_dump()}


def invariant_history(conn: sqlite3.Connection, session_id: str) -> list[dict]:
    rows = conn.execute(
        "SELECT id, version_hash, previous_hash, origin, created_at"
        " FROM session_invariant_assignments WHERE session_id = ? ORDER BY id",
        (session_id,),
    ).fetchall()
    return [dict(row) for row in rows]


def record_invariant_check(
    conn: sqlite3.Connection,
    session_id: str,
    invariant_hash: str,
    target: str,
    outcome: str,
    rule_ids: list[str],
    explanation: str,
    alternative: str | None,
    seq: int,
) -> None:
    conn.execute(
        "INSERT INTO invariant_checks"
        " (session_id, invariant_hash, target, outcome, rule_ids_json, explanation,"
        " alternative, seq, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            session_id,
            invariant_hash,
            target,
            outcome,
            json.dumps(rule_ids, ensure_ascii=False),
            explanation,
            alternative,
            seq,
            _now(),
        ),
    )
    conn.commit()


def list_invariant_checks(conn: sqlite3.Connection, session_id: str) -> list[dict]:
    rows = conn.execute(
        "SELECT id, invariant_hash, target, outcome, rule_ids_json, explanation,"
        " alternative, seq, created_at FROM invariant_checks"
        " WHERE session_id = ? ORDER BY id",
        (session_id,),
    ).fetchall()
    result = []
    for row in rows:
        item = dict(row)
        item["rule_ids"] = json.loads(item.pop("rule_ids_json"))
        result.append(item)
    return result


def set_session_invariants(
    conn: sqlite3.Connection,
    session_id: str,
    invariants: InvariantSet,
    *,
    expected_hash: str,
    origin: str = "edit",
    replan_task: bool = False,
) -> str:
    """Atomically move a session pointer when the browser edited the same version."""
    session = get_session(conn, session_id)
    if session is None:
        return "missing"
    if session["run_id"] is not None:
        return "run"
    current_hash = session["invariant_hash"]
    if current_hash != expected_hash:
        return "conflict"
    task_row = conn.execute(
        "SELECT * FROM tasks WHERE session_id = ?"
        " AND stage NOT IN ('done', 'cancelled')",
        (session_id,),
    ).fetchone()
    replanned = None
    if task_row is not None:
        if not replan_task:
            return "replan"
        replanned = transition(_task_state(task_row), "planning", "user")
    digest = invariants.hash
    if digest == current_hash and replanned is None:
        return "unchanged"
    with conn:
        save_invariant_version(conn, invariants)
        cursor = conn.execute(
            "UPDATE sessions SET invariant_hash = ?, updated_at = ?"
            " WHERE id = ? AND invariant_hash = ?",
            (digest, _now(), session_id, expected_hash),
        )
        if cursor.rowcount != 1:
            raise sqlite3.IntegrityError("invariant version changed concurrently")
        conn.execute(
            "INSERT INTO session_invariant_assignments"
            " (session_id, version_hash, previous_hash, origin, created_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (session_id, digest, current_hash, origin, _now()),
        )
        if task_row is not None and replanned is not None:
            now = _now()
            conn.execute(
                "UPDATE tasks SET stage = 'planning', paused = 0, current_step = NULL,"
                " verdict = NULL, steps_json = '[]', criteria_json = '[]',"
                " plan_text = NULL, plan_error = NULL, revision = ?, result_revision = ?,"
                " approved_plan_hash = NULL, approved_invariant_hash = NULL,"
                " verdict_result_revision = NULL, updated_at = ?, closed_at = NULL"
                " WHERE id = ?",
                (replanned.revision, replanned.result_revision, now, task_row["id"]),
            )
            _journal(
                conn,
                task_row["id"],
                replanned,
                kind="invariants",
                from_stage=task_row["stage"],
                actor="user",
                reason="изменён набор инвариантов",
                seq=_next_seq(conn, session_id),
            )
    return "updated"


# --- named presets --------------------------------------------------------


def _preset(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "name": row["preset_name"],
        "is_default": bool(row["is_default"]),
        "spec_hash": row["spec_hash"],
        "spec": _load_spec_dict(row["spec_json"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def _preset_select() -> str:
    return (
        "SELECT p.id, p.name AS preset_name, p.spec_hash, p.is_default,"
        "       p.created_at, p.updated_at, a.spec_json"
        "  FROM agent_presets p JOIN agent_specs a ON a.hash = p.spec_hash"
    )


def create_preset(
    conn: sqlite3.Connection, name: str, spec: AgentSpec, is_default: bool = False
) -> dict:
    preset_id = _new_id()
    now = _now()
    try:
        with conn:
            digest = _save_spec(conn, spec)
            if is_default:
                conn.execute("UPDATE agent_presets SET is_default = 0 WHERE is_default = 1")
            conn.execute(
                "INSERT INTO agent_presets (id, name, spec_hash, is_default, created_at, updated_at)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (preset_id, name, digest, int(is_default), now, now),
            )
    except Exception:
        conn.rollback()
        raise
    preset = get_preset(conn, preset_id)
    assert preset is not None
    return preset


def get_preset(conn: sqlite3.Connection, preset_id: str) -> dict | None:
    row = conn.execute(_preset_select() + " WHERE p.id = ?", (preset_id,)).fetchone()
    return _preset(row) if row else None


def get_default_preset(conn: sqlite3.Connection) -> dict | None:
    row = conn.execute(_preset_select() + " WHERE p.is_default = 1").fetchone()
    return _preset(row) if row else None


def list_presets(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        _preset_select() + " ORDER BY p.is_default DESC, LOWER(p.name), p.created_at"
    ).fetchall()
    return [_preset(row) for row in rows]


def update_preset(
    conn: sqlite3.Connection, preset_id: str, name: str, spec: AgentSpec
) -> dict | None:
    if get_preset(conn, preset_id) is None:
        return None
    now = _now()
    try:
        with conn:
            digest = _save_spec(conn, spec)
            conn.execute(
                "UPDATE agent_presets SET name = ?, spec_hash = ?, updated_at = ? WHERE id = ?",
                (name, digest, now, preset_id),
            )
    except Exception:
        conn.rollback()
        raise
    return get_preset(conn, preset_id)


def set_default_preset(conn: sqlite3.Connection, preset_id: str) -> dict | None:
    if get_preset(conn, preset_id) is None:
        return None
    now = _now()
    with conn:
        conn.execute("UPDATE agent_presets SET is_default = 0 WHERE is_default = 1")
        conn.execute(
            "UPDATE agent_presets SET is_default = 1, updated_at = ? WHERE id = ?",
            (now, preset_id),
        )
    return get_preset(conn, preset_id)


def delete_preset(conn: sqlite3.Connection, preset_id: str) -> bool:
    cursor = conn.execute("DELETE FROM agent_presets WHERE id = ?", (preset_id,))
    conn.commit()
    return cursor.rowcount > 0


# --- user profiles ---------------------------------------------------------


def save_user_profile_version(conn: sqlite3.Connection, profile: UserProfile) -> str:
    digest = profile.hash
    conn.execute(
        "INSERT OR IGNORE INTO user_profile_versions (hash, body_json, created_at)"
        " VALUES (?, ?, ?)",
        (digest, profile.canonical_json(), _now()),
    )
    return digest


def get_user_profile_version(
    conn: sqlite3.Connection, digest: str
) -> UserProfile | None:
    row = conn.execute(
        "SELECT body_json FROM user_profile_versions WHERE hash = ?", (digest,)
    ).fetchone()
    return UserProfile.model_validate_json(row["body_json"]) if row else None


def _user_profile(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "name": row["name"],
        "version_hash": row["version_hash"],
        "is_default": bool(row["is_default"]),
        "profile": json.loads(row["body_json"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


_USER_PROFILE_SELECT = (
    "SELECT u.id, u.name, u.version_hash, u.is_default, u.created_at,"
    "       u.updated_at, v.body_json"
    "  FROM user_profiles u"
    "  JOIN user_profile_versions v ON v.hash = u.version_hash"
)


def create_user_profile(
    conn: sqlite3.Connection, profile: UserProfile, is_default: bool = False
) -> dict:
    """The pointer's name is the profile's own name, so it has one source."""
    profile_id = _new_id()
    now = _now()
    with conn:
        digest = save_user_profile_version(conn, profile)
        if is_default:
            conn.execute("UPDATE user_profiles SET is_default = 0 WHERE is_default = 1")
        conn.execute(
            "INSERT INTO user_profiles"
            " (id, name, version_hash, is_default, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (profile_id, profile.name, digest, int(is_default), now, now),
        )
    created = get_user_profile(conn, profile_id)
    assert created is not None
    return created


def get_user_profile(conn: sqlite3.Connection, profile_id: str) -> dict | None:
    row = conn.execute(
        _USER_PROFILE_SELECT + " WHERE u.id = ?", (profile_id,)
    ).fetchone()
    return _user_profile(row) if row else None


def get_default_user_profile(conn: sqlite3.Connection) -> dict | None:
    row = conn.execute(_USER_PROFILE_SELECT + " WHERE u.is_default = 1").fetchone()
    return _user_profile(row) if row else None


def list_user_profiles(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        _USER_PROFILE_SELECT + " ORDER BY u.is_default DESC, LOWER(u.name), u.created_at"
    ).fetchall()
    return [_user_profile(row) for row in rows]


def update_user_profile(
    conn: sqlite3.Connection, profile_id: str, profile: UserProfile
) -> dict | None:
    if get_user_profile(conn, profile_id) is None:
        return None
    with conn:
        digest = save_user_profile_version(conn, profile)
        conn.execute(
            "UPDATE user_profiles SET name = ?, version_hash = ?, updated_at = ?"
            " WHERE id = ?",
            (profile.name, digest, _now(), profile_id),
        )
    return get_user_profile(conn, profile_id)


class _PointerMoved(Exception):
    pass


def update_user_profile_about(
    conn: sqlite3.Connection, profile_id: str, about: str
) -> tuple[str, dict | None]:
    """Saves `about` as a new version. The pointer moves only if nobody moved it
    since the read, so a concurrent PUT is never silently overwritten; on a
    conflict the new version is rolled back with the pointer."""
    current = get_user_profile(conn, profile_id)
    if current is None:
        return "missing", None
    profile = UserProfile.model_validate({**current["profile"], "about": about})
    try:
        with conn:
            digest = save_user_profile_version(conn, profile)
            cursor = conn.execute(
                "UPDATE user_profiles SET version_hash = ?, updated_at = ?"
                " WHERE id = ? AND version_hash = ?",
                (digest, _now(), profile_id, current["version_hash"]),
            )
            if cursor.rowcount == 0:
                raise _PointerMoved
            # Read before the commit: the answer is this edit, not a later PUT.
            row = conn.execute(
                _USER_PROFILE_SELECT + " WHERE u.id = ?", (profile_id,)
            ).fetchone()
    except _PointerMoved:
        exists = conn.execute(
            "SELECT 1 FROM user_profiles WHERE id = ?", (profile_id,)
        ).fetchone()
        return ("changed" if exists else "missing"), None
    return "updated", _user_profile(row)


def set_default_user_profile(conn: sqlite3.Connection, profile_id: str) -> dict | None:
    if get_user_profile(conn, profile_id) is None:
        return None
    with conn:
        conn.execute("UPDATE user_profiles SET is_default = 0 WHERE is_default = 1")
        conn.execute(
            "UPDATE user_profiles SET is_default = 1, updated_at = ? WHERE id = ?",
            (_now(), profile_id),
        )
    return get_user_profile(conn, profile_id)


def delete_user_profile(conn: sqlite3.Connection, profile_id: str) -> bool:
    """Sessions lose the pointer; versions stay for the ledger rows citing them."""
    cursor = conn.execute("DELETE FROM user_profiles WHERE id = ?", (profile_id,))
    conn.commit()
    return cursor.rowcount > 0


def set_session_user_profile(
    conn: sqlite3.Connection, session_id: str, profile_id: str | None
) -> bool:
    cursor = conn.execute(
        "UPDATE sessions SET user_profile_id = ?, updated_at = ? WHERE id = ?",
        (profile_id, _now(), session_id),
    )
    conn.commit()
    return cursor.rowcount > 0


def create_session(
    conn: sqlite3.Connection,
    spec: AgentSpec,
    title: str | None = None,
    user_profile_id: str | None = None,
) -> str:
    digest = save_spec(conn, spec)
    session_id = _new_id()
    now = _now()
    # A session joins its memory space at birth, not at the first long-term
    # write: that is what makes a fresh session in the same space start out
    # already knowing the profile stored there.
    scope_id = (
        ensure_scope(conn, spec.memory.longterm_scope)
        if spec.memory.longterm_enabled
        else None
    )
    invariants = InvariantSet()
    with conn:
        invariant_hash = save_invariant_version(conn, invariants)
        conn.execute(
            "INSERT INTO sessions"
            " (id, spec_hash, title, memory_scope_id, user_profile_id, invariant_hash,"
            "  status, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, 'active', ?, ?)",
            (session_id, digest, title, scope_id, user_profile_id, invariant_hash, now, now),
        )
        conn.execute(
            "INSERT INTO session_invariant_assignments"
            " (session_id, version_hash, previous_hash, origin, created_at)"
            " VALUES (?, ?, NULL, 'create', ?)",
            (session_id, invariant_hash, now),
        )
    return session_id


def get_session(conn: sqlite3.Connection, session_id: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM sessions WHERE id = ?", (session_id,)
    ).fetchone()


def list_sessions(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        "SELECT s.id, s.title, s.created_at, s.updated_at,"
        "       s.parent_id, s.fork_seq, a.spec_json,"
        "       t.stage AS task_stage, t.paused AS task_paused,"
        "       s.user_profile_id, u.name AS user_profile_name, s.invariant_hash,"
        "       (SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id) AS messages,"
        "       (SELECT COALESCE(SUM(cost_usd), 0) FROM ledger l"
        "          WHERE l.session_id = s.id) AS cost_usd"
        "  FROM sessions s JOIN agent_specs a ON a.hash = s.spec_hash"
        "  LEFT JOIN user_profiles u ON u.id = s.user_profile_id"
        "  LEFT JOIN tasks t ON t.session_id = s.id"
        "   AND t.stage NOT IN ('done', 'cancelled')"
        " ORDER BY s.updated_at DESC"
    ).fetchall()
    return [
        {
            "id": r["id"],
            "title": r["title"],
            "created_at": r["created_at"],
            "updated_at": r["updated_at"],
            "spec": _load_spec_dict(r["spec_json"]),
            "executor_kind": _load_spec(r["spec_json"]).tools.executor,
            "task_stage": r["task_stage"],
            "task_paused": bool(r["task_paused"]),
            "parent_id": r["parent_id"],
            "fork_seq": r["fork_seq"],
            "user_profile_id": r["user_profile_id"],
            "user_profile_name": r["user_profile_name"],
            "invariant_hash": r["invariant_hash"],
            "messages": r["messages"],
            "cost_usd": round(r["cost_usd"], 6),
        }
        for r in rows
    ]


def save_checkpoint(
    conn: sqlite3.Connection, session_id: str, seq: int, name: str
) -> bool:
    """Create or rename the one checkpoint attached to a transcript row."""
    exists = conn.execute(
        "SELECT 1 FROM messages WHERE session_id = ? AND seq = ?", (session_id, seq)
    ).fetchone()
    if exists is None:
        return False
    conn.execute(
        "INSERT INTO checkpoints (session_id, seq, name) VALUES (?, ?, ?)"
        " ON CONFLICT(session_id, seq) DO UPDATE SET name = excluded.name",
        (session_id, seq, name),
    )
    conn.commit()
    return True


def list_checkpoints(conn: sqlite3.Connection, session_id: str) -> list[dict]:
    rows = conn.execute(
        "SELECT session_id, seq, name FROM checkpoints"
        " WHERE session_id = ? ORDER BY seq",
        (session_id,),
    ).fetchall()
    return [dict(row) for row in rows]


def fork_session(
    conn: sqlite3.Connection,
    session_id: str,
    seq: int,
    title: str | None = None,
    *,
    spec: AgentSpec | None = None,
) -> str | None:
    """Copy a session snapshot through `seq`; spend and later memory stay behind."""
    source = get_session(conn, session_id)
    if source is None:
        return None
    messages = conn.execute(
        "SELECT id, seq, role, content, created_at FROM messages"
        " WHERE session_id = ? AND seq <= ? ORDER BY seq",
        (session_id, seq),
    ).fetchall()
    if not messages or messages[-1]["seq"] != seq:
        return None

    fork_id = _new_id()
    now = _now()
    try:
        with conn:
            # A replay installs its configuration with the branch. Updating it
            # afterwards could extract memory again from the copied dialogue.
            spec_hash = _save_spec(conn, spec) if spec is not None else source["spec_hash"]
            conn.execute(
                "INSERT INTO sessions"
                " (id, spec_hash, parent_id, fork_seq, title,"
                "  chars_per_token, memory_scope_id, memory_covered_seq,"
                "  user_profile_id, invariant_hash, status, created_at, updated_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?)",
                (
                    fork_id,
                    spec_hash,
                    session_id,
                    seq,
                    title or f"ветка от #{seq}",
                    source["chars_per_token"],
                    source["memory_scope_id"],
                    # The branch inherits coverage only up to its own last
                    # message; anything the router read later is not its past.
                    min(int(source["memory_covered_seq"]), seq),
                    # The branch talks to the same person: it follows the
                    # pointer, not the version current at the fork point.
                    source["user_profile_id"],
                    source["invariant_hash"],
                    now,
                    now,
                ),
            )
            conn.execute(
                "INSERT INTO session_invariant_assignments"
                " (session_id, version_hash, previous_hash, origin, created_at)"
                " VALUES (?, ?, NULL, 'fork', ?)",
                (fork_id, source["invariant_hash"], now),
            )
            copied_ids: dict[int, int] = {}
            for message in messages:
                cursor = conn.execute(
                    "INSERT INTO messages (session_id, seq, role, content, created_at)"
                    " VALUES (?, ?, ?, ?, ?)",
                    (
                        fork_id,
                        message["seq"],
                        message["role"],
                        message["content"],
                        message["created_at"],
                    ),
                )
                copied_ids[message["id"]] = int(cursor.lastrowid or 0)

            calls = conn.execute(
                "SELECT t.message_id, t.tool, t.arguments_json, t.status, t.result,"
                "       t.exit_code, t.duration_ms, t.created_at"
                "  FROM tool_calls t JOIN messages m ON m.id = t.message_id"
                " WHERE t.session_id = ? AND m.seq <= ? ORDER BY m.seq",
                (session_id, seq),
            ).fetchall()
            for call in calls:
                conn.execute(
                    "INSERT INTO tool_calls"
                    " (id, session_id, message_id, tool, arguments_json, status, result,"
                    "  exit_code, duration_ms, created_at)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        _new_id(),
                        fork_id,
                        copied_ids[call["message_id"]],
                        call["tool"],
                        call["arguments_json"],
                        call["status"],
                        call["result"],
                        call["exit_code"],
                        call["duration_ms"],
                        call["created_at"],
                    ),
                )

            conn.execute(
                "INSERT INTO session_summaries"
                " (session_id, version, covered_seq, message_count, summary,"
                "  summary_tokens, created_at)"
                " SELECT ?, version, covered_seq, message_count, summary,"
                "        summary_tokens, created_at FROM session_summaries"
                " WHERE session_id = ? AND covered_seq <= ? ORDER BY version",
                (fork_id, session_id, seq),
            )
            # Working memory is part of the snapshot; the long-term layer is
            # not — it belongs to the scope, which the branch simply joins.
            conn.execute(
                "INSERT INTO memory_versions"
                " (layer, session_id, version, covered_seq, origin, items_json,"
                "  tokens, created_at)"
                " SELECT 'working', ?, version, covered_seq, origin, items_json,"
                "        tokens, created_at FROM memory_versions"
                " WHERE layer = 'working' AND session_id = ? AND covered_seq <= ?"
                " ORDER BY version",
                (fork_id, session_id, seq),
            )
            _copy_task(conn, session_id, fork_id, seq, now)
    except Exception:
        conn.rollback()
        raise
    return fork_id


def _copy_task(
    conn: sqlite3.Connection, session_id: str, fork_id: str, seq: int, now: str
) -> None:
    """Give the branch the task as it stood at the fork point.

    Restored from the journal, not from the parent's current row: a branch
    taken before step 3 must not start with step 3 already done.
    """
    row = conn.execute(
        "SELECT tr.state_json FROM task_transitions tr"
        "  JOIN tasks t ON t.id = tr.task_id"
        " WHERE t.session_id = ? AND tr.seq <= ? ORDER BY tr.id DESC LIMIT 1",
        (session_id, seq),
    ).fetchone()
    if row is None:
        return
    state = TaskState.from_dict(json.loads(row["state_json"]))
    if state.is_terminal:
        return
    fork_invariant_hash = conn.execute(
        "SELECT invariant_hash FROM sessions WHERE id = ?", (fork_id,)
    ).fetchone()["invariant_hash"]
    if (
        state.stage in ("execution", "validation")
        and state.approved_invariant_hash != fork_invariant_hash
    ):
        # The branch inherits the session's current rules, not a historical
        # approval for a different set. Replanning is the only safe restore.
        state = transition(state, "planning", "user")
    task_id = _new_id()
    conn.execute(
        "INSERT INTO tasks (id, session_id, stage, paused, current_step, verdict,"
        " revision, result_revision, approved_plan_hash, approved_invariant_hash,"
        " verdict_result_revision, steps_json, criteria_json, plan_text, plan_error,"
        " created_at, updated_at, covered_seq)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            task_id,
            fork_id,
            state.stage,
            int(state.paused),
            state.current_step,
            state.verdict,
            state.revision,
            state.result_revision,
            state.approved_plan_hash,
            state.approved_invariant_hash,
            state.verdict_result_revision,
            json.dumps([item.to_dict() for item in state.steps], ensure_ascii=False),
            json.dumps(list(state.criteria), ensure_ascii=False),
            state.plan_text or None,
            state.plan_error,
            now,
            now,
            seq,
        ),
    )
    _journal(
        conn, task_id, state, kind="fork", from_stage=None, actor="user",
        reason=f"ветка от #{seq}", seq=seq,
    )


def delete_session(conn: sqlite3.Connection, session_id: str) -> bool:
    cursor = conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
    conn.commit()
    return cursor.rowcount > 0


def update_session_spec(
    conn: sqlite3.Connection, session_id: str, spec: AgentSpec
) -> str:
    session = get_session(conn, session_id)
    if session is None:
        return "missing"
    if session["run_id"] is not None:
        return "run"
    with conn:
        digest = _save_spec(conn, spec)
        conn.execute(
            "UPDATE sessions SET spec_hash = ?, updated_at = ? WHERE id = ?",
            (digest, _now(), session_id),
        )
    return "updated"


def load_messages(conn: sqlite3.Connection, session_id: str) -> list[dict]:
    """The transcript in order, tool rows filled in from `tool_calls`.

    A tool row in `messages` carries no data of its own — it only marks where
    the call belongs in the dialogue. Everything about the call lives once, in
    `tool_calls`, and is joined back in here.
    """
    rows = conn.execute(
        "SELECT m.id, m.seq, m.role, m.content, m.created_at, t.id AS call_id,"
        "       t.tool, t.arguments_json, t.status, t.result, t.exit_code,"
        "       t.duration_ms"
        "  FROM messages m LEFT JOIN tool_calls t ON t.message_id = m.id"
        " WHERE m.session_id = ? ORDER BY m.seq",
        (session_id,),
    ).fetchall()
    entries = []
    for row in rows:
        if row["role"] != "tool":
            entries.append(
                {
                    "id": row["id"],
                    "seq": row["seq"],
                    "role": row["role"],
                    "content": row["content"],
                    "created_at": row["created_at"],
                }
            )
            continue
        entries.append(
            {
                "id": row["id"],
                "seq": row["seq"],
                "role": "tool",
                "created_at": row["created_at"],
                "call_id": row["call_id"],
                "tool": row["tool"],
                "arguments_json": row["arguments_json"],
                "status": row["status"],
                "content": row["result"] or "",
                "exit_code": row["exit_code"],
                "duration_ms": row["duration_ms"],
            }
        )
    return entries


def list_summaries(conn: sqlite3.Connection, session_id: str) -> list[dict]:
    rows = conn.execute(
        "SELECT id, version, covered_seq, message_count, summary, summary_tokens, created_at"
        " FROM session_summaries WHERE session_id = ? ORDER BY version",
        (session_id,),
    ).fetchall()
    return [dict(row) for row in rows]


def latest_summary(conn: sqlite3.Connection, session_id: str) -> dict | None:
    row = conn.execute(
        "SELECT id, version, covered_seq, message_count, summary AS text,"
        " summary_tokens AS tokens, created_at FROM session_summaries"
        " WHERE session_id = ? ORDER BY version DESC LIMIT 1",
        (session_id,),
    ).fetchone()
    return dict(row) if row else None


def save_summary(
    conn: sqlite3.Connection,
    session_id: str,
    *,
    covered_seq: int,
    message_count: int,
    text: str,
    tokens: int,
) -> dict:
    row = conn.execute(
        "SELECT COALESCE(MAX(version), 0) + 1 AS version FROM session_summaries"
        " WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    version = int(row["version"])
    now = _now()
    cursor = conn.execute(
        "INSERT INTO session_summaries"
        " (session_id, version, covered_seq, message_count, summary, summary_tokens, created_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?)",
        (session_id, version, covered_seq, message_count, text, tokens, now),
    )
    conn.commit()
    return {
        "id": int(cursor.lastrowid or 0),
        "version": version,
        "covered_seq": covered_seq,
        "message_count": message_count,
        "summary": text,
        "summary_tokens": tokens,
        "created_at": now,
    }


def ensure_scope(conn: sqlite3.Connection, name: str) -> str:
    """Long-term memory is addressed by scope name; sessions join it by id."""
    row = conn.execute(
        "SELECT id FROM memory_scopes WHERE name = ?", (name,)
    ).fetchone()
    if row is not None:
        return str(row["id"])
    scope_id = _new_id()
    conn.execute(
        "INSERT INTO memory_scopes (id, name, created_at) VALUES (?, ?, ?)",
        (scope_id, name, _now()),
    )
    conn.commit()
    return scope_id


def scope_sessions(conn: sqlite3.Connection, name: str) -> int:
    """How many sessions joined this space — the layer's reach, in one number."""
    row = conn.execute(
        "SELECT COUNT(*) AS total FROM sessions s"
        " JOIN memory_scopes m ON m.id = s.memory_scope_id WHERE m.name = ?",
        (name,),
    ).fetchone()
    return int(row["total"])


def attach_scope(conn: sqlite3.Connection, session_id: str, name: str) -> str:
    scope_id = ensure_scope(conn, name)
    conn.execute(
        "UPDATE sessions SET memory_scope_id = ? WHERE id = ?", (scope_id, session_id)
    )
    conn.commit()
    return scope_id


def _owner(
    session_id: str | None, scope_id: str | None, user_profile_id: str | None = None
) -> tuple[str, str]:
    """Exactly one owner: a session or a scope, or a user profile.

    Memory versions have only the first two; a profile owns router candidates
    alone, its versions live in `user_profile_versions`.
    """
    owners = [
        (column, value)
        for column, value in (
            ("session_id", session_id),
            ("scope_id", scope_id),
            ("user_profile_id", user_profile_id),
        )
        if value is not None
    ]
    if len(owners) != 1:
        raise ValueError(
            "Запись памяти принадлежит ровно одному владельцу:"
            " сессии, пространству или профилю"
        )
    return owners[0]


def list_memory(
    conn: sqlite3.Connection,
    layer: str,
    *,
    session_id: str | None = None,
    scope_id: str | None = None,
) -> list[dict]:
    column, owner = _owner(session_id, scope_id)
    rows = conn.execute(
        "SELECT id, version, covered_seq, origin, items_json, tokens, created_at"
        f" FROM memory_versions WHERE layer = ? AND {column} = ? ORDER BY version",
        (layer, owner),
    ).fetchall()
    return [_memory_row(row) for row in rows]


def latest_memory(
    conn: sqlite3.Connection,
    layer: str,
    *,
    session_id: str | None = None,
    scope_id: str | None = None,
) -> dict | None:
    column, owner = _owner(session_id, scope_id)
    row = conn.execute(
        "SELECT id, version, covered_seq, origin, items_json, tokens, created_at"
        f" FROM memory_versions WHERE layer = ? AND {column} = ?"
        " ORDER BY version DESC LIMIT 1",
        (layer, owner),
    ).fetchone()
    return _memory_row(row) if row else None


def save_memory(
    conn: sqlite3.Connection,
    layer: str,
    *,
    session_id: str | None = None,
    scope_id: str | None = None,
    covered_seq: int,
    items: dict[str, dict],
    tokens: int,
    origin: str,
) -> dict:
    """Append a new immutable version; earlier ones stay readable."""
    column, owner = _owner(session_id, scope_id)
    row = conn.execute(
        "SELECT COALESCE(MAX(version), 0) + 1 AS version FROM memory_versions"
        f" WHERE layer = ? AND {column} = ?",
        (layer, owner),
    ).fetchone()
    cursor = conn.execute(
        "INSERT INTO memory_versions"
        f" (layer, {column}, version, covered_seq, origin, items_json, tokens, created_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            layer,
            owner,
            int(row["version"]),
            covered_seq,
            origin,
            json.dumps(items, ensure_ascii=False, sort_keys=True),
            tokens,
            _now(),
        ),
    )
    conn.commit()
    result = latest_memory(conn, layer, session_id=session_id, scope_id=scope_id)
    assert result is not None and result["id"] == int(cursor.lastrowid or 0)
    return result


def set_memory_covered(conn: sqlite3.Connection, session_id: str, seq: int) -> None:
    """Move the router watermark forward; it never walks back on its own."""
    conn.execute(
        "UPDATE sessions SET memory_covered_seq = MAX(memory_covered_seq, ?)"
        " WHERE id = ?",
        (seq, session_id),
    )
    conn.commit()


def save_proposals(
    conn: sqlite3.Connection,
    layer: str,
    *,
    session_id: str | None = None,
    scope_id: str | None = None,
    user_profile_id: str | None = None,
    changes: dict[str, dict],
    source_seq: int,
) -> list[dict]:
    """Park router output for a human; one live candidate per key."""
    column, owner = _owner(session_id, scope_id, user_profile_id)
    if not changes:
        return []
    now = _now()
    with conn:
        for key, change in changes.items():
            conn.execute(
                f"DELETE FROM memory_proposals WHERE layer = ? AND {column} = ?"
                " AND key = ?",
                (layer, owner, key),
            )
            conn.execute(
                "INSERT INTO memory_proposals"
                f" (layer, {column}, key, value, kind, source_seq, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    layer,
                    owner,
                    key,
                    str(change.get("value") or ""),
                    str(change.get("kind") or ""),
                    source_seq,
                    now,
                ),
            )
    return list_proposals(
        conn,
        layer,
        session_id=session_id,
        scope_id=scope_id,
        user_profile_id=user_profile_id,
    )


def list_proposals(
    conn: sqlite3.Connection,
    layer: str,
    *,
    session_id: str | None = None,
    scope_id: str | None = None,
    user_profile_id: str | None = None,
) -> list[dict]:
    column, owner = _owner(session_id, scope_id, user_profile_id)
    rows = conn.execute(
        "SELECT id, layer, key, value, kind, source_seq, created_at"
        f" FROM memory_proposals WHERE layer = ? AND {column} = ?"
        " ORDER BY created_at, id",
        (layer, owner),
    ).fetchall()
    return [dict(row) for row in rows]


def get_proposal(conn: sqlite3.Connection, proposal_id: int) -> dict | None:
    row = conn.execute(
        "SELECT id, layer, session_id, scope_id, user_profile_id, key, value, kind,"
        " source_seq, created_at FROM memory_proposals WHERE id = ?",
        (proposal_id,),
    ).fetchone()
    return dict(row) if row else None


def delete_proposal(conn: sqlite3.Connection, proposal_id: int) -> None:
    conn.execute("DELETE FROM memory_proposals WHERE id = ?", (proposal_id,))
    conn.commit()


def _memory_row(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "version": row["version"],
        "covered_seq": row["covered_seq"],
        "origin": row["origin"],
        "items": json.loads(row["items_json"]),
        "tokens": row["tokens"],
        "created_at": row["created_at"],
    }


def append_message(
    conn: sqlite3.Connection, session_id: str, role: str, content: str
) -> int:
    row = conn.execute(
        "SELECT COALESCE(MAX(seq), 0) + 1 AS seq FROM messages WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    seq = row["seq"]
    now = _now()
    cursor = conn.execute(
        "INSERT INTO messages (session_id, seq, role, content, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (session_id, seq, role, content, now),
    )
    conn.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (now, session_id))
    conn.commit()
    return int(cursor.lastrowid or 0)


def drop_last_user_message(conn: sqlite3.Connection, session_id: str) -> bool:
    """Remove the trailing question of a turn that produced no answer.

    Only when it is still the last row: once a tool has run, the question is
    part of a dialogue the agent kept in memory, and a retry must not rewrite it.
    """
    row = conn.execute(
        "SELECT id, role FROM messages WHERE session_id = ? ORDER BY seq DESC LIMIT 1",
        (session_id,),
    ).fetchone()
    if row is None or row["role"] != "user":
        return False
    conn.execute("DELETE FROM messages WHERE id = ?", (row["id"],))
    conn.commit()
    return True


def record_tool_call(
    conn: sqlite3.Connection,
    session_id: str,
    call: ToolCall,
    result: ToolResult,
    status: str,
) -> None:
    """Store a finished call: one transcript marker plus the call itself."""
    message_id = append_message(conn, session_id, "tool", "")
    conn.execute(
        "INSERT INTO tool_calls (id, session_id, message_id, tool, arguments_json,"
        " status, result, exit_code, duration_ms, created_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            _new_id(),
            session_id,
            message_id,
            call.tool,
            call.arguments_json,
            status,
            result.output,
            result.exit_code,
            result.duration_ms,
            _now(),
        ),
    )
    conn.commit()


def record_usage(
    conn: sqlite3.Connection,
    session_id: str,
    model: str,
    usage: Usage,
    latency_ms: int | None = None,
    *,
    spec_hash: str,
    message_id: int | None = None,
    estimate: dict | None = None,
    context: dict | None = None,
    kind: str = "chat",
    user_profile_hash: str | None = None,
    invariant_hash: str | None = None,
) -> None:
    """Record the configuration snapshots owned by this model call."""
    estimate = estimate or {}
    context = context or {}
    if invariant_hash is None:
        row = conn.execute(
            "SELECT invariant_hash FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        invariant_hash = row["invariant_hash"] if row else None
    conn.execute(
        "INSERT INTO ledger (session_id, spec_hash, message_id, model, kind, input_tokens, output_tokens,"
        " reasoning_tokens, cost_usd, role_estimate_tokens, history_estimate_tokens,"
        " request_estimate_tokens, input_estimate_tokens, context_estimate_tokens,"
        " context_input_tokens, context_output_tokens, context_limit_tokens,"
        " context_limit_source, dropped_messages, latency_ms, created_at,"
        " user_profile_hash, invariant_hash)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            session_id,
            spec_hash,
            message_id,
            model,
            kind,
            usage.tokens_in,
            usage.tokens_out,
            usage.reasoning,
            usage.cost,
            estimate.get("role_tokens", 0),
            estimate.get("history_tokens", 0),
            estimate.get("request_tokens", 0),
            estimate.get("input_tokens", 0),
            context.get("estimated_tokens", 0),
            context.get("input_tokens", 0),
            context.get("output_tokens", 0),
            context.get("limit_tokens"),
            context.get("limit_source"),
            context.get("dropped_messages", 0),
            latency_ms,
            _now(),
            user_profile_hash,
            invariant_hash,
        ),
    )
    chars_per_token = context.get("chars_per_token")
    if chars_per_token is not None:
        conn.execute(
            "UPDATE sessions SET chars_per_token = ? WHERE id = ?",
            (chars_per_token, session_id),
        )
    conn.commit()


def session_totals(conn: sqlite3.Connection, session_id: str) -> dict:
    row = conn.execute(
        "SELECT COUNT(*) AS requests,"
        "       COALESCE(SUM(input_tokens), 0) AS tokens_in,"
        "       COALESCE(SUM(output_tokens), 0) AS tokens_out,"
        "       COALESCE(SUM(reasoning_tokens), 0) AS reasoning,"
        "       COALESCE(SUM(cost_usd), 0) AS cost_usd"
        "  FROM ledger WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    return dict(row)


def session_usage(conn: sqlite3.Connection, session_id: str) -> list[dict]:
    """Per-turn estimates and facts for the token panel."""
    rows = conn.execute(
        "SELECT id, spec_hash, message_id, model, kind, input_tokens, output_tokens, reasoning_tokens, cost_usd,"
        " role_estimate_tokens, history_estimate_tokens, request_estimate_tokens,"
        " input_estimate_tokens, context_estimate_tokens, context_input_tokens,"
        " context_output_tokens, context_limit_tokens, context_limit_source,"
        " dropped_messages, latency_ms, created_at, user_profile_hash, invariant_hash FROM ledger"
        " WHERE session_id = ? ORDER BY id",
        (session_id,),
    ).fetchall()
    return [
        {
            **dict(row),
            "cost_usd": round(row["cost_usd"], 6),
            "estimate_delta_tokens": row["input_tokens"] - row["input_estimate_tokens"],
        }
        for row in rows
    ]


# --- matrix runs -----------------------------------------------------------


def create_run(
    conn: sqlite3.Connection,
    *,
    name: str | None,
    prompt: str,
    repeats: int,
    budget_usd: float | None,
    estimated_cost_usd: float,
    specs: list[AgentSpec],
    config_names: list[str] | None = None,
    user_profile_hashes: list[str | None] | None = None,
    invariant_sets: list[InvariantSet] | None = None,
) -> str:
    """Persist the exact configuration matrix before any paid call starts.

    A configuration cites a profile version, not the profile: editing the
    profile later must not change what this run is recorded to have sent.
    """
    run_id = _new_id()
    conn.execute(
        "INSERT INTO runs (id, name, prompt, repeats, budget_usd, estimated_cost_usd, status, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, 'running', ?)",
        (run_id, name, prompt, repeats, budget_usd, estimated_cost_usd, _now()),
    )
    for index, spec in enumerate(specs):
        digest = save_spec(conn, spec)
        invariant_hash = (
            save_invariant_version(conn, invariant_sets[index])
            if invariant_sets is not None
            else None
        )
        conn.execute(
            "INSERT INTO run_configurations"
            " (run_id, config_index, name, spec_hash, user_profile_hash, invariant_hash)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (
                run_id,
                index,
                config_names[index] if config_names else f"Конфигурация {index + 1}",
                digest,
                user_profile_hashes[index] if user_profile_hashes else None,
                invariant_hash,
            ),
        )
    conn.commit()
    return run_id


def record_run_usage(
    conn: sqlite3.Connection,
    run_id: str,
    spec: AgentSpec,
    model: str,
    usage: Usage,
    latency_ms: int,
    user_profile_hash: str | None = None,
    invariant_hash: str | None = None,
    kind: str = "chat",
) -> None:
    digest = _save_spec(conn, spec)
    conn.execute(
        "INSERT INTO ledger (run_id, spec_hash, model, kind, input_tokens, output_tokens,"
        " reasoning_tokens, cost_usd, latency_ms, created_at, user_profile_hash, invariant_hash)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            run_id,
            digest,
            model,
            kind,
            usage.tokens_in,
            usage.tokens_out,
            usage.reasoning,
            usage.cost,
            latency_ms,
            _now(),
            user_profile_hash,
            invariant_hash,
        ),
    )
    conn.commit()


def record_run_result(
    conn: sqlite3.Connection,
    run_id: str,
    config_index: int,
    repeat_index: int,
    status: str,
    result: dict,
) -> None:
    conn.execute(
        "INSERT INTO run_results (id, run_id, config_index, repeat_index, status, result_json, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            _new_id(),
            run_id,
            config_index,
            repeat_index,
            status,
            json.dumps(result, ensure_ascii=False, sort_keys=True),
            _now(),
        ),
    )
    conn.commit()


def finish_run(conn: sqlite3.Connection, run_id: str, status: str) -> None:
    conn.execute(
        "UPDATE runs SET status = ?, finished_at = ? WHERE id = ?",
        (status, _now(), run_id),
    )
    conn.commit()


def get_run(conn: sqlite3.Connection, run_id: str) -> dict | None:
    run = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    if run is None:
        return None
    configs = conn.execute(
        "SELECT c.config_index, c.name, s.hash, s.spec_json, c.user_profile_hash,"
        "       v.body_json AS user_profile_json, c.invariant_hash,"
        "       i.body_json AS invariant_json"
        "  FROM run_configurations c"
        "  JOIN agent_specs s ON s.hash = c.spec_hash"
        "  LEFT JOIN user_profile_versions v ON v.hash = c.user_profile_hash"
        "  LEFT JOIN invariant_versions i ON i.hash = c.invariant_hash"
        " WHERE c.run_id = ? ORDER BY c.config_index",
        (run_id,),
    ).fetchall()
    results = conn.execute(
        "SELECT config_index, repeat_index, status, result_json FROM run_results "
        "WHERE run_id = ? ORDER BY config_index, repeat_index",
        (run_id,),
    ).fetchall()
    totals = conn.execute(
        "SELECT COUNT(*) AS requests, COALESCE(SUM(input_tokens), 0) AS tokens_in, "
        "COALESCE(SUM(output_tokens), 0) AS tokens_out, "
        "COALESCE(SUM(reasoning_tokens), 0) AS reasoning, "
        "COALESCE(SUM(cost_usd), 0) AS cost_usd FROM ledger WHERE run_id = ?",
        (run_id,),
    ).fetchone()
    return {
        "id": run["id"],
        "name": run["name"],
        "prompt": run["prompt"],
        "repeats": run["repeats"],
        "budget_usd": run["budget_usd"],
        "estimated_cost_usd": run["estimated_cost_usd"],
        "status": run["status"],
        "created_at": run["created_at"],
        "finished_at": run["finished_at"],
        "configs": [
            {
                "index": row["config_index"],
                "hash": row["hash"],
                "name": row["name"],
                "spec": _load_spec_dict(row["spec_json"]),
                "user_profile": (
                    {
                        "hash": row["user_profile_hash"],
                        "profile": json.loads(row["user_profile_json"]),
                    }
                    if row["user_profile_hash"]
                    else None
                ),
                "invariants": (
                    {
                        "hash": row["invariant_hash"],
                        **json.loads(row["invariant_json"]),
                    }
                    if row["invariant_hash"]
                    else None
                ),
            }
            for row in configs
        ],
        "results": [
            {"config_index": row["config_index"], "repeat_index": row["repeat_index"], "status": row["status"], **json.loads(row["result_json"])}
            for row in results
        ],
        "totals": dict(totals),
    }


# --- tasks ------------------------------------------------------------------


def _task(row: sqlite3.Row) -> dict:
    state = _task_state(row)
    return {
        "id": row["id"],
        "session_id": row["session_id"],
        **state.to_dict(),
        "covered_seq": row["covered_seq"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "closed_at": row["closed_at"],
    }


def _task_state(row: sqlite3.Row) -> TaskState:
    return TaskState(
        stage=row["stage"],
        paused=bool(row["paused"]),
        steps=tuple(Step.from_dict(item) for item in json.loads(row["steps_json"])),
        current_step=row["current_step"],
        criteria=tuple(json.loads(row["criteria_json"])),
        verdict=row["verdict"],
        plan_text=row["plan_text"] or "",
        plan_error=row["plan_error"],
        revision=int(row["revision"]),
        result_revision=int(row["result_revision"]),
        approved_plan_hash=row["approved_plan_hash"],
        approved_invariant_hash=row["approved_invariant_hash"],
        verdict_result_revision=row["verdict_result_revision"],
    )


def _next_seq(conn: sqlite3.Connection, session_id: str) -> int:
    row = conn.execute(
        "SELECT COALESCE(MAX(seq), 0) AS seq FROM messages WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    return int(row["seq"])


def active_task(conn: sqlite3.Connection, session_id: str) -> dict | None:
    row = conn.execute(
        "SELECT * FROM tasks WHERE session_id = ?"
        " AND stage NOT IN ('done', 'cancelled')",
        (session_id,),
    ).fetchone()
    return _task(row) if row is not None else None


def get_task(conn: sqlite3.Connection, task_id: str) -> dict | None:
    row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
    return _task(row) if row is not None else None


def session_tasks(conn: sqlite3.Connection, session_id: str) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM tasks WHERE session_id = ? ORDER BY created_at", (session_id,)
    ).fetchall()
    return [_task(row) for row in rows]


def create_task(
    conn: sqlite3.Connection, session_id: str, *, actor: str = "user", seq: int | None = None
) -> dict | None:
    """Start a task on the session; `None` means one is already active."""
    if active_task(conn, session_id) is not None:
        return None
    if get_session(conn, session_id) is None:
        return None
    task_id = _new_id()
    now = _now()
    state = TaskState()
    with conn:
        conn.execute(
            "INSERT INTO tasks (id, session_id, stage, paused, current_step, verdict,"
            " revision, result_revision, steps_json, criteria_json, plan_text, plan_error,"
            " created_at, updated_at, covered_seq)"
            " VALUES (?, ?, 'planning', 0, NULL, NULL, 0, 0, '[]', '[]', NULL, NULL, ?, ?, ?)",
            (task_id, session_id, now, now, _next_seq(conn, session_id)),
        )
        _journal(
            conn,
            task_id,
            state,
            kind="start",
            from_stage=None,
            actor=actor,
            reason=None,
            seq=seq if seq is not None else _next_seq(conn, session_id),
        )
    return get_task(conn, task_id)


def _journal(
    conn: sqlite3.Connection,
    task_id: str,
    state: TaskState,
    *,
    kind: str,
    from_stage: str | None,
    actor: str,
    reason: str | None,
    seq: int,
) -> None:
    conn.execute(
        "INSERT INTO task_transitions"
        " (task_id, kind, from_stage, to_stage, paused, actor, reason, seq,"
        "  state_json, created_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            task_id,
            kind,
            from_stage,
            state.stage,
            int(state.paused),
            actor,
            reason,
            seq,
            json.dumps(state.to_dict(), ensure_ascii=False),
            _now(),
        ),
    )


def _store_task(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
    state: TaskState,
    *,
    kind: str,
    actor: str,
    reason: str | None,
    seq: int | None,
) -> dict:
    """One write path: the row, the journal entry and the closing stamp."""
    now = _now()
    closed = now if state.is_terminal else None
    with conn:
        cursor = conn.execute(
            "UPDATE tasks SET stage = ?, paused = ?, current_step = ?, verdict = ?,"
            " revision = ?, result_revision = ?, approved_plan_hash = ?,"
            " approved_invariant_hash = ?, verdict_result_revision = ?,"
            " steps_json = ?, criteria_json = ?, plan_text = ?, plan_error = ?,"
            " updated_at = ?, closed_at = ? WHERE id = ? AND revision = ?",
            (
                state.stage,
                int(state.paused),
                state.current_step,
                state.verdict,
                state.revision,
                state.result_revision,
                state.approved_plan_hash,
                state.approved_invariant_hash,
                state.verdict_result_revision,
                json.dumps([item.to_dict() for item in state.steps], ensure_ascii=False),
                json.dumps(list(state.criteria), ensure_ascii=False),
                state.plan_text or None,
                state.plan_error,
                now,
                closed,
                row["id"],
                row["revision"],
            ),
        )
        if cursor.rowcount != 1:
            raise TaskError("stale_task", "Состояние задачи уже изменилось; обновите экран")
        _journal(
            conn,
            row["id"],
            state,
            kind=kind,
            from_stage=row["stage"],
            actor=actor,
            reason=reason,
            seq=seq if seq is not None else _next_seq(conn, row["session_id"]),
        )
    return get_task(conn, row["id"])


def _task_row(conn: sqlite3.Connection, task_id: str) -> sqlite3.Row:
    row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
    if row is None:
        raise TaskError("missing", "Задачи нет")
    return row


def apply_transition(
    conn: sqlite3.Connection,
    task_id: str,
    to_stage: str,
    *,
    actor: str = "user",
    force: bool = False,
    failed_criteria: tuple[str, ...] = (),
    reason: str | None = None,
    seq: int | None = None,
    expected_revision: int | None = None,
    expected_invariant_hash: str | None = None,
) -> dict:
    """Move the task; `TaskError` names the guard that refused."""
    row = _task_row(conn, task_id)
    if expected_revision is not None and int(row["revision"]) != expected_revision:
        raise TaskError("stale_task", "Состояние задачи уже изменилось; обновите экран")
    invariant_hash = conn.execute(
        "SELECT invariant_hash FROM sessions WHERE id = ?", (row["session_id"],)
    ).fetchone()["invariant_hash"]
    if expected_invariant_hash is not None and expected_invariant_hash != invariant_hash:
        raise TaskError("stale_invariants", "Инварианты уже изменились; обновите экран")
    current = _task_state(row)
    if (
        current.stage in ("execution", "validation")
        and to_stage not in ("planning", "cancelled")
        and current.approved_invariant_hash != invariant_hash
    ):
        raise TaskError(
            "stale_approval",
            "Утверждение относится к другой версии инвариантов; вернитесь в планирование",
        )
    state = transition(
        current,
        to_stage,
        actor,
        force=force,
        failed_criteria=failed_criteria,
        invariant_hash=invariant_hash,
    )
    moved = _store_task(conn, row, state, kind="stage", actor=actor, reason=reason, seq=seq)
    if row["stage"] == "planning" and to_stage == "execution":
        return set_task_covered(conn, task_id, _next_seq(conn, row["session_id"]))
    return moved


def set_task_covered(conn: sqlite3.Connection, task_id: str, seq: int) -> dict:
    """Move the tracker watermark forward; it never walks back."""
    with conn:
        conn.execute(
            "UPDATE tasks SET covered_seq = MAX(covered_seq, ?) WHERE id = ?",
            (seq, task_id),
        )
    return get_task(conn, task_id)


def set_task_paused(
    conn: sqlite3.Connection,
    task_id: str,
    paused: bool,
    *,
    actor: str = "user",
    reason: str | None = None,
    seq: int | None = None,
    expected_revision: int | None = None,
) -> dict:
    row = _task_row(conn, task_id)
    if expected_revision is not None and int(row["revision"]) != expected_revision:
        raise TaskError("stale_task", "Состояние задачи уже изменилось; обновите экран")
    state = set_paused(_task_state(row), paused, actor)
    return _store_task(conn, row, state, kind="pause", actor=actor, reason=reason, seq=seq)


def save_task_plan(
    conn: sqlite3.Connection,
    task_id: str,
    plan_text: str,
    *,
    actor: str = "user",
    seq: int | None = None,
    expected_revision: int | None = None,
) -> dict:
    row = _task_row(conn, task_id)
    if expected_revision is not None and int(row["revision"]) != expected_revision:
        raise TaskError("stale_task", "Состояние задачи уже изменилось; обновите экран")
    state = plan_state(_task_state(row), plan_text)
    return _store_task(
        conn, row, state, kind="plan", actor=actor, reason=state.plan_error, seq=seq
    )


def update_task_step(
    conn: sqlite3.Connection,
    task_id: str,
    index: int,
    *,
    status: str | None = None,
    title: str | None = None,
    note: str | None = None,
    actor: str = "user",
    seq: int | None = None,
    expected_revision: int | None = None,
) -> dict:
    row = _task_row(conn, task_id)
    if expected_revision is not None and int(row["revision"]) != expected_revision:
        raise TaskError("stale_task", "Состояние задачи уже изменилось; обновите экран")
    state = set_step(
        _task_state(row), index, actor, status=status, title=title, note=note
    )
    return _store_task(
        conn, row, state, kind="step", actor=actor, reason=note, seq=seq
    )


def set_task_verdict(
    conn: sqlite3.Connection,
    task_id: str,
    verdict: str | None,
    *,
    actor: str = "tracker",
    reason: str | None = None,
    seq: int | None = None,
    expected_revision: int | None = None,
) -> dict:
    row = _task_row(conn, task_id)
    if expected_revision is not None and int(row["revision"]) != expected_revision:
        raise TaskError("stale_task", "Состояние задачи уже изменилось; обновите экран")
    state = set_verdict(_task_state(row), verdict, actor)
    return _store_task(conn, row, state, kind="verdict", actor=actor, reason=reason, seq=seq)


def record_task_rejection(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    actor: str,
    reason: str,
    seq: int | None = None,
) -> None:
    """Append a refused decision without changing the authoritative state."""
    row = _task_row(conn, task_id)
    state = _task_state(row)
    with conn:
        _journal(
            conn,
            task_id,
            state,
            kind="rejected",
            from_stage=row["stage"],
            actor=actor,
            reason=reason,
            seq=seq if seq is not None else _next_seq(conn, row["session_id"]),
        )


_TRANSITION_COLUMNS = (
    "id, task_id, kind, from_stage, to_stage, paused, actor, reason, seq, created_at"
)


def _transition(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "task_id": row["task_id"],
        "kind": row["kind"],
        "from_stage": row["from_stage"],
        "to_stage": row["to_stage"],
        "paused": bool(row["paused"]),
        "actor": row["actor"],
        "reason": row["reason"],
        "seq": row["seq"],
        "created_at": row["created_at"],
    }


def list_task_transitions(conn: sqlite3.Connection, task_id: str) -> list[dict]:
    rows = conn.execute(
        f"SELECT {_TRANSITION_COLUMNS} FROM task_transitions WHERE task_id = ? ORDER BY id",
        (task_id,),
    ).fetchall()
    return [_transition(row) for row in rows]


def session_task_transitions(conn: sqlite3.Connection, session_id: str) -> list[dict]:
    """The journal of every task of the session, closed ones included.

    The transcript marks transitions where they happened, so a task that ended
    keeps its marks: the active task alone would erase them on `done`.
    """
    rows = conn.execute(
        f"SELECT {_TRANSITION_COLUMNS} FROM task_transitions"
        " WHERE task_id IN (SELECT id FROM tasks WHERE session_id = ?)"
        " ORDER BY id",
        (session_id,),
    ).fetchall()
    return [_transition(row) for row in rows]


def task_state_at(conn: sqlite3.Connection, task_id: str, seq: int) -> TaskState | None:
    """The task as it stood on transcript row `seq` — what a branch copies."""
    row = conn.execute(
        "SELECT state_json FROM task_transitions WHERE task_id = ? AND seq <= ?"
        " ORDER BY id DESC LIMIT 1",
        (task_id, seq),
    ).fetchone()
    if row is None:
        return None
    return TaskState.from_dict(json.loads(row["state_json"]))
