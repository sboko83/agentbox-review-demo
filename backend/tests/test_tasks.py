import pytest

from agentbox.core.spec import AgentSpec
from agentbox.core.task_machine import TaskError
from agentbox.store import repository as repo
from agentbox.store.db import init_db, table_names

PLAN = "1. Написать a.txt\n2. Написать b.txt\nПроверка:\n- оба файла на месте"


@pytest.fixture
def db(tmp_path):
    connection = init_db(tmp_path / "tasks.db")
    yield connection
    connection.close()


@pytest.fixture
def session(db):
    session_id = repo.create_session(db, AgentSpec())
    for index in range(1, 5):
        repo.append_message(db, session_id, "user", f"сообщение {index}")
    return session_id


def test_a_session_holds_one_active_task_at_a_time(db, session):
    task = repo.create_task(db, session)

    assert task["stage"] == "planning"
    assert task["paused"] is False
    assert repo.create_task(db, session) is None
    assert repo.active_task(db, session)["id"] == task["id"]

    repo.apply_transition(db, task["id"], "cancelled")

    assert repo.active_task(db, session) is None
    assert repo.create_task(db, session)["id"] != task["id"]
    assert len(repo.session_tasks(db, session)) == 2


def test_the_full_cycle_is_persisted_with_its_journal(db, session):
    task = repo.create_task(db, session)
    task_id = task["id"]

    repo.save_task_plan(db, task_id, PLAN)
    repo.apply_transition(db, task_id, "execution")
    repo.update_task_step(db, task_id, 1, status="done", note="файл создан")
    repo.update_task_step(db, task_id, 2, status="done")
    repo.apply_transition(db, task_id, "validation", actor="tracker")
    repo.set_task_verdict(db, task_id, "pass")
    closed = repo.apply_transition(db, task_id, "done")

    assert closed["stage"] == "done"
    assert closed["closed_at"] is not None
    assert closed["criteria"] == ["оба файла на месте"]
    assert [step["status"] for step in closed["steps"]] == ["done", "done"]

    journal = repo.list_task_transitions(db, task_id)
    assert [entry["kind"] for entry in journal] == [
        "start", "plan", "stage", "step", "step", "stage", "verdict", "stage",
    ]
    assert [entry["actor"] for entry in journal][5] == "tracker"
    assert all(entry["seq"] == 4 for entry in journal)


def test_the_session_journal_outlives_the_task_it_belongs_to(db, session):
    """The transcript marks transitions, so a closed task keeps its marks."""
    first = repo.create_task(db, session)["id"]
    repo.apply_transition(db, first, "cancelled")
    second = repo.create_task(db, session)["id"]
    repo.set_task_paused(db, second, True)

    journal = repo.session_task_transitions(db, session)

    assert [entry["kind"] for entry in journal] == ["start", "stage", "start", "pause"]
    assert [entry["task_id"] for entry in journal] == [first, first, second, second]
    assert repo.active_task(db, session)["id"] == second


def test_an_unparsed_plan_is_kept_as_a_draft_and_blocks_execution(db, session):
    task = repo.create_task(db, session)

    draft = repo.save_task_plan(db, task["id"], "сначала подумаю")

    assert draft["steps"] == []
    assert "не разобран" in draft["plan_error"]
    with pytest.raises(TaskError) as refused:
        repo.apply_transition(db, task["id"], "execution")
    assert refused.value.code == "no_steps"


def test_a_refused_transition_leaves_the_row_and_the_journal_untouched(db, session):
    task = repo.create_task(db, session)
    repo.save_task_plan(db, task["id"], PLAN)
    repo.apply_transition(db, task["id"], "execution")
    before = repo.get_task(db, task["id"])

    with pytest.raises(TaskError):
        repo.apply_transition(db, task["id"], "done")
    with pytest.raises(TaskError):
        repo.update_task_step(db, task["id"], 9, status="done")

    after = repo.get_task(db, task["id"])
    assert (after["stage"], after["steps"]) == (before["stage"], before["steps"])
    assert len(repo.list_task_transitions(db, task["id"])) == 3


def test_pause_and_resume_return_the_same_stage_after_a_reopen(db, session, tmp_path):
    task = repo.create_task(db, session)
    repo.save_task_plan(db, task["id"], PLAN)
    repo.apply_transition(db, task["id"], "execution")

    paused = repo.set_task_paused(db, task["id"], True)
    assert paused["paused"] is True
    assert paused["expected_action"]["text"] == "продолжить задачу"
    with pytest.raises(TaskError):
        repo.set_task_paused(db, task["id"], True)
    db.close()

    reopened = init_db(tmp_path / "tasks.db")
    restored = repo.active_task(reopened, session)
    assert restored["stage"] == "execution" and restored["paused"] is True

    resumed = repo.set_task_paused(reopened, task["id"], False)
    assert resumed["stage"] == "execution"
    assert resumed["current_step"] == 1
    assert resumed["expected_action"]["text"] == "выполнить шаг 1 «Написать a.txt»"
    reopened.close()


def test_the_state_at_a_transcript_row_is_what_a_branch_copies(db, session):
    task = repo.create_task(db, session, seq=1)
    repo.save_task_plan(db, task["id"], PLAN, seq=2)
    repo.apply_transition(db, task["id"], "execution", seq=2)
    repo.update_task_step(db, task["id"], 1, status="done", seq=4)

    early = repo.task_state_at(db, task["id"], 1)
    fork = repo.task_state_at(db, task["id"], 3)
    latest = repo.task_state_at(db, task["id"], 9)

    assert early.stage == "planning" and early.steps == ()
    assert fork.stage == "execution"
    assert fork.current_step == 1 and fork.step(1).status == "in_progress"
    assert latest.step(1).status == "done" and latest.current_step == 2
    assert repo.task_state_at(db, task["id"], 0) is None


def test_a_branch_gets_the_task_as_it_stood_at_the_fork_point(db, session):
    task = repo.create_task(db, session, seq=1)
    repo.save_task_plan(db, task["id"], PLAN, seq=1)
    repo.apply_transition(db, task["id"], "execution", seq=1)
    repo.update_task_step(db, task["id"], 1, status="done", seq=4)

    fork_id = repo.fork_session(db, session, 2)

    branch = repo.active_task(db, fork_id)
    assert branch["id"] != task["id"]
    assert branch["stage"] == "execution"
    # Step 1 was finished after the fork point, so the branch still owes it.
    assert [step["status"] for step in branch["steps"]] == ["in_progress", "todo"]
    assert branch["current_step"] == 1
    assert repo.list_task_transitions(db, branch["id"])[0]["kind"] == "fork"
    assert repo.get_task(db, task["id"])["steps"][0]["status"] == "done"


def test_a_branch_taken_before_the_task_started_has_none(db, session):
    repo.create_task(db, session, seq=3)

    fork_id = repo.fork_session(db, session, 2)

    assert repo.active_task(db, fork_id) is None


def test_a_branch_of_a_closed_task_starts_without_one(db, session):
    task = repo.create_task(db, session, seq=1)
    repo.apply_transition(db, task["id"], "cancelled", seq=2)

    fork_id = repo.fork_session(db, session, 3)

    assert repo.active_task(db, fork_id) is None
    assert repo.session_tasks(db, fork_id) == []


def test_deleting_a_session_takes_its_tasks_and_journal_with_it(db, session):
    task = repo.create_task(db, session)
    repo.save_task_plan(db, task["id"], PLAN)

    repo.delete_session(db, session)

    assert repo.get_task(db, task["id"]) is None
    assert repo.list_task_transitions(db, task["id"]) == []


def test_the_upgrade_adds_the_task_tables_and_keeps_the_data(tmp_path):
    path = tmp_path / "old.db"
    connection = init_db(path)
    session_id = repo.create_session(connection, AgentSpec())
    repo.append_message(connection, session_id, "user", "привет")
    connection.execute("PRAGMA user_version = 13")
    connection.execute("DROP TABLE task_transitions")
    connection.execute("DROP TABLE tasks")
    connection.commit()
    connection.close()

    upgraded = init_db(path)

    assert upgraded.execute("PRAGMA user_version").fetchone()[0] == 19
    assert {"tasks", "task_transitions"} <= set(table_names(upgraded))
    assert repo.get_session(upgraded, session_id) is not None
    assert repo.create_task(upgraded, session_id)["stage"] == "planning"
    upgraded.close()


def test_a_session_left_in_planning_mode_reopens_as_a_task(tmp_path):
    path = tmp_path / "v14.db"
    connection = init_db(path)
    session_id = repo.create_session(connection, AgentSpec())
    repo.append_message(connection, session_id, "user", "составь план")
    plain_id = repo.create_session(connection, AgentSpec(), title="обычная")
    connection.execute(
        "ALTER TABLE sessions ADD COLUMN planning_mode TEXT NOT NULL DEFAULT 'execution'"
    )
    connection.execute("ALTER TABLE sessions ADD COLUMN plan TEXT")
    connection.execute(
        "UPDATE sessions SET planning_mode = 'planning', plan = ? WHERE id = ?",
        (PLAN, session_id),
    )
    connection.execute("PRAGMA user_version = 14")
    connection.commit()
    connection.close()

    upgraded = init_db(path)

    columns = {
        row["name"] for row in upgraded.execute("PRAGMA table_info(sessions)").fetchall()
    }
    assert {"planning_mode", "plan"} & columns == set()
    assert repo.load_messages(upgraded, session_id)[0]["content"] == "составь план"

    task = repo.active_task(upgraded, session_id)
    assert task["stage"] == "planning"
    assert task["plan_text"] == PLAN
    assert [step["title"] for step in task["steps"]] == [
        "Написать a.txt",
        "Написать b.txt",
    ]
    assert repo.list_task_transitions(upgraded, task["id"])[0]["kind"] == "migration"
    assert repo.active_task(upgraded, plain_id) is None
    # The rebuilt table keeps its children and its index.
    assert repo.delete_session(upgraded, session_id) is True
    assert repo.load_messages(upgraded, session_id) == []
    assert repo.get_task(upgraded, task["id"]) is None
    upgraded.close()
