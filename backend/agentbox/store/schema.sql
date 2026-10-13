-- agentbox schema, version 17.
-- Applied on startup; `PRAGMA user_version` guards re-application.

CREATE TABLE IF NOT EXISTS agent_specs (
    hash        TEXT PRIMARY KEY,   -- sha256 of the canonical spec JSON
    spec_json   TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS agent_presets (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,
    spec_hash   TEXT NOT NULL REFERENCES agent_specs(hash),
    is_default  INTEGER NOT NULL DEFAULT 0 CHECK (is_default IN (0, 1)),
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_agent_presets_default
    ON agent_presets (is_default) WHERE is_default = 1;

-- The user profile repeats the preset pair: immutable versions addressed by
-- content hash and a named pointer that sessions follow.
CREATE TABLE IF NOT EXISTS user_profile_versions (
    hash        TEXT PRIMARY KEY,   -- sha256 of the canonical profile JSON
    body_json   TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS user_profiles (
    id           TEXT PRIMARY KEY,
    name         TEXT NOT NULL UNIQUE,
    version_hash TEXT NOT NULL REFERENCES user_profile_versions(hash),
    is_default   INTEGER NOT NULL DEFAULT 0 CHECK (is_default IN (0, 1)),
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_user_profiles_default
    ON user_profiles (is_default) WHERE is_default = 1;

-- Session invariants are human-owned. Versions never change; a session points
-- at the complete set it currently uses.
CREATE TABLE IF NOT EXISTS invariant_versions (
    hash        TEXT PRIMARY KEY,
    body_json   TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    id                 TEXT PRIMARY KEY,
    name               TEXT,
    prompt             TEXT NOT NULL,
    repeats            INTEGER NOT NULL DEFAULT 1,
    budget_usd         REAL,
    estimated_cost_usd REAL,
    status             TEXT NOT NULL,
    created_at         TEXT NOT NULL,
    finished_at        TEXT
);

-- Memory layers that outlive the context window. A working-memory version
-- belongs to one session; a long-term version belongs to a scope shared by
-- every session that joined it — that is the whole difference between them.
CREATE TABLE IF NOT EXISTS memory_scopes (
    id         TEXT PRIMARY KEY,
    name       TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    id             TEXT PRIMARY KEY,
    spec_hash      TEXT NOT NULL REFERENCES agent_specs(hash),
    run_id         TEXT REFERENCES runs(id) ON DELETE CASCADE,
    parent_id      TEXT REFERENCES sessions(id) ON DELETE SET NULL,
    fork_seq       INTEGER,
    title          TEXT,
    chars_per_token REAL NOT NULL DEFAULT 3.0, -- calibrated estimate for this session
    memory_scope_id TEXT REFERENCES memory_scopes(id) ON DELETE SET NULL,
    -- How far the router has read this transcript. Coverage belongs to the
    -- session, not to a layer version: a long-term version is shared by every
    -- session of its scope and its own `covered_seq` counts another dialogue.
    memory_covered_seq INTEGER NOT NULL DEFAULT 0,
    -- A pointer, not a version: editing the profile applies from the next call.
    user_profile_id TEXT REFERENCES user_profiles(id) ON DELETE SET NULL,
    invariant_hash TEXT REFERENCES invariant_versions(hash),
    status         TEXT NOT NULL,
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS session_invariant_assignments (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    version_hash  TEXT NOT NULL REFERENCES invariant_versions(hash),
    previous_hash TEXT REFERENCES invariant_versions(hash),
    origin        TEXT NOT NULL, -- create | edit | migration | fork
    created_at    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_session_invariant_assignments_session
    ON session_invariant_assignments (session_id, id);

CREATE TABLE IF NOT EXISTS invariant_checks (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id     TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    invariant_hash TEXT NOT NULL REFERENCES invariant_versions(hash),
    target         TEXT NOT NULL, -- request | response | tool
    outcome        TEXT NOT NULL, -- allow | conflict | error
    rule_ids_json  TEXT NOT NULL DEFAULT '[]',
    explanation    TEXT NOT NULL,
    alternative    TEXT,
    seq            INTEGER NOT NULL,
    created_at     TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_invariant_checks_session
    ON invariant_checks (session_id, id);

CREATE TABLE IF NOT EXISTS messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    seq        INTEGER NOT NULL,
    role       TEXT NOT NULL,       -- user | assistant | tool
    -- A `tool` row is only a marker of position in the dialogue; the call
    -- itself lives once, in tool_calls, linked back by message_id.
    content    TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (session_id, seq)
);

CREATE TABLE IF NOT EXISTS session_summaries (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id     TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    version        INTEGER NOT NULL,
    covered_seq    INTEGER NOT NULL,
    message_count  INTEGER NOT NULL,
    summary        TEXT NOT NULL,
    summary_tokens INTEGER NOT NULL,
    created_at     TEXT NOT NULL,
    UNIQUE (session_id, version)
);

CREATE TABLE IF NOT EXISTS memory_versions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    layer       TEXT NOT NULL CHECK (layer IN ('working', 'longterm')),
    session_id  TEXT REFERENCES sessions(id) ON DELETE CASCADE,
    scope_id    TEXT REFERENCES memory_scopes(id) ON DELETE CASCADE,
    version     INTEGER NOT NULL,
    covered_seq INTEGER NOT NULL DEFAULT 0,
    origin      TEXT NOT NULL,   -- router | manual | promote | migration
    items_json  TEXT NOT NULL,   -- key -> {value, kind, origin, seq}
    tokens      INTEGER NOT NULL,
    created_at  TEXT NOT NULL,
    CHECK ((session_id IS NULL) <> (scope_id IS NULL))
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_memory_versions_session
    ON memory_versions (session_id, layer, version) WHERE session_id IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS idx_memory_versions_scope
    ON memory_versions (scope_id, layer, version) WHERE scope_id IS NOT NULL;

-- Router output waiting for a human: memory that outlives the dialogue is
-- only written once somebody accepted it. One row per key — a newer candidate
-- for the same key replaces the older one instead of queueing behind it.
CREATE TABLE IF NOT EXISTS memory_proposals (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    layer      TEXT NOT NULL CHECK (layer IN ('working', 'longterm', 'profile')),
    session_id TEXT REFERENCES sessions(id) ON DELETE CASCADE,
    scope_id   TEXT REFERENCES memory_scopes(id) ON DELETE CASCADE,
    user_profile_id TEXT REFERENCES user_profiles(id) ON DELETE CASCADE,
    key        TEXT NOT NULL,
    value      TEXT NOT NULL,   -- empty value proposes forgetting the key
    kind       TEXT NOT NULL,
    source_seq INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    CHECK ((session_id IS NOT NULL) + (scope_id IS NOT NULL)
           + (user_profile_id IS NOT NULL) = 1)
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_memory_proposals_session
    ON memory_proposals (session_id, layer, key) WHERE session_id IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS idx_memory_proposals_scope
    ON memory_proposals (scope_id, layer, key) WHERE scope_id IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS idx_memory_proposals_profile
    ON memory_proposals (user_profile_id, key) WHERE user_profile_id IS NOT NULL;

-- A task is a state machine attached to a session: at most one is active.
-- Its stage decides which tools the agent gets, so the row is the authority,
-- never the chat text.
CREATE TABLE IF NOT EXISTS tasks (
    id            TEXT PRIMARY KEY,
    session_id    TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    stage         TEXT NOT NULL CHECK (stage IN
                      ('planning', 'execution', 'validation', 'done', 'cancelled')),
    -- Pause is a flag, not a stage: it does not double the number of states
    -- and resuming returns the very stage the task stopped on.
    paused        INTEGER NOT NULL DEFAULT 0 CHECK (paused IN (0, 1)),
    current_step  INTEGER,
    verdict       TEXT CHECK (verdict IN ('pass', 'fail')),
    revision      INTEGER NOT NULL DEFAULT 0,
    result_revision INTEGER NOT NULL DEFAULT 0,
    approved_plan_hash TEXT,
    approved_invariant_hash TEXT REFERENCES invariant_versions(hash),
    verdict_result_revision INTEGER,
    steps_json    TEXT NOT NULL DEFAULT '[]',
    criteria_json TEXT NOT NULL DEFAULT '[]',
    plan_text     TEXT,
    plan_error    TEXT,            -- why the draft plan did not parse into steps
    -- How far the tracker has read the transcript: it pays per call, so a
    -- turn it already looked at is never sent again.
    covered_seq   INTEGER NOT NULL DEFAULT 0,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    closed_at     TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_tasks_active
    ON tasks (session_id) WHERE stage NOT IN ('done', 'cancelled');

-- Append-only journal of everything that moved the task, stamped with the
-- transcript row it happened on. `state_json` is the snapshot after the event,
-- so a branch restores the task as it was at `fork_seq` instead of copying the
-- parent's current state.
CREATE TABLE IF NOT EXISTS task_transitions (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id    TEXT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
    kind       TEXT NOT NULL,   -- start | stage | pause | step | plan | verdict
    from_stage TEXT,
    to_stage   TEXT NOT NULL,
    paused     INTEGER NOT NULL DEFAULT 0 CHECK (paused IN (0, 1)),
    actor      TEXT NOT NULL,   -- user | tracker
    reason     TEXT,
    seq        INTEGER NOT NULL,
    state_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_task_transitions_task
    ON task_transitions (task_id, id);

CREATE TABLE IF NOT EXISTS checkpoints (
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    seq        INTEGER NOT NULL,
    name       TEXT NOT NULL,
    PRIMARY KEY (session_id, seq)
);

CREATE TABLE IF NOT EXISTS tool_calls (
    id             TEXT PRIMARY KEY,
    session_id     TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    message_id     INTEGER REFERENCES messages(id) ON DELETE SET NULL,
    tool           TEXT NOT NULL,
    arguments_json TEXT NOT NULL,
    status         TEXT NOT NULL,   -- pending | rejected | ok | error
    result         TEXT,
    exit_code      INTEGER,
    duration_ms    INTEGER,
    created_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ledger (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id       TEXT REFERENCES sessions(id) ON DELETE CASCADE,
    run_id           TEXT REFERENCES runs(id) ON DELETE CASCADE,
    spec_hash        TEXT NOT NULL REFERENCES agent_specs(hash),
    user_profile_hash TEXT REFERENCES user_profile_versions(hash),
    invariant_hash TEXT REFERENCES invariant_versions(hash),
    message_id       INTEGER REFERENCES messages(id) ON DELETE SET NULL,
    model            TEXT NOT NULL,
    kind             TEXT NOT NULL DEFAULT 'chat', -- chat | summary | memory | task
    input_tokens     INTEGER NOT NULL DEFAULT 0,
    output_tokens    INTEGER NOT NULL DEFAULT 0,
    reasoning_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd         REAL NOT NULL DEFAULT 0,
    role_estimate_tokens    INTEGER NOT NULL DEFAULT 0,
    history_estimate_tokens INTEGER NOT NULL DEFAULT 0,
    request_estimate_tokens INTEGER NOT NULL DEFAULT 0,
    input_estimate_tokens   INTEGER NOT NULL DEFAULT 0,
    context_estimate_tokens INTEGER NOT NULL DEFAULT 0,
    context_input_tokens    INTEGER NOT NULL DEFAULT 0,
    context_output_tokens   INTEGER NOT NULL DEFAULT 0,
    context_limit_tokens    INTEGER,
    context_limit_source    TEXT,
    dropped_messages        INTEGER NOT NULL DEFAULT 0,
    latency_ms       INTEGER,
    created_at       TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_session   ON messages (session_id, seq);
CREATE INDEX IF NOT EXISTS idx_summaries_session  ON session_summaries (session_id, version);
CREATE INDEX IF NOT EXISTS idx_tool_calls_session ON tool_calls (session_id);
CREATE INDEX IF NOT EXISTS idx_sessions_run       ON sessions (run_id);
CREATE INDEX IF NOT EXISTS idx_ledger_session     ON ledger (session_id);
CREATE INDEX IF NOT EXISTS idx_ledger_run         ON ledger (run_id);

-- Phase 4 keeps a run's immutable configurations and its final results. The
-- ledger remains the single source of actual call spend, including the judge.
CREATE TABLE IF NOT EXISTS run_configurations (
    run_id       TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    config_index INTEGER NOT NULL,
    name         TEXT NOT NULL,
    spec_hash    TEXT NOT NULL REFERENCES agent_specs(hash),
    user_profile_hash TEXT REFERENCES user_profile_versions(hash),
    invariant_hash TEXT REFERENCES invariant_versions(hash),
    PRIMARY KEY (run_id, config_index)
);

CREATE TABLE IF NOT EXISTS run_results (
    id           TEXT PRIMARY KEY,
    run_id       TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    config_index INTEGER NOT NULL,
    repeat_index INTEGER NOT NULL,
    status       TEXT NOT NULL,
    result_json  TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    UNIQUE (run_id, config_index, repeat_index)
);

CREATE INDEX IF NOT EXISTS idx_run_results_run ON run_results (run_id, config_index);
