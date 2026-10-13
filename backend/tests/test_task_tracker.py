"""The tracker marks what a turn did; the machine decides what lands."""

import json
from dataclasses import replace

import pytest

from agentbox.api.routes.sessions import _apply_tracker
from agentbox.core.ledger import Ledger
from agentbox.core.invariants import InvariantRule, InvariantSet
from agentbox.core.task_machine import Step, TaskState, plan_state, transition
from agentbox.core.task_tracker import (
    TaskTracker,
    TrackerResult,
    TrackerStep,
    tracker_runs,
)
from agentbox.llm.stub import StubLLM
from agentbox.store import repository as repo

from test_sessions_api import client, sse_events  # noqa: F401 - shared fixture

PLAN = "1. Написать файл\nПроверка:\n- файл на месте"
INVARIANT_HASH = "0" * 64
HISTORY = [
    {"seq": 1, "role": "user", "content": "Сделай первый шаг."},
    {"seq": 2, "role": "assistant", "content": "Файл написан."},
]


def _executing() -> TaskState:
    state = plan_state(TaskState(), PLAN)
    return transition(state, "execution", invariant_hash=INVARIANT_HASH)


def _tracker_answer(**fields) -> str:
    return json.dumps({"step": None, "verdict": None, "propose": None, **fields})


# --- the tracker on its own --------------------------------------------------


@pytest.mark.parametrize(
    "state",
    [
        None,
        TaskState(),
        TaskState(stage="execution", paused=True, steps=(Step(1, "шаг"),)),
        TaskState(stage="done"),
    ],
)
async def test_the_tracker_watches_only_a_task_in_work(state):
    assert tracker_runs(state) is False
    if state is not None:
        stub = StubLLM(_tracker_answer())
        assert await TaskTracker("gpt-5-nano", 64).observe(
            stub, state, HISTORY, Ledger()
        ) is None
        assert stub.requests == []


async def test_a_mark_is_parsed_and_the_word_null_means_nothing():
    stub = StubLLM(
        _tracker_answer(
            step={"index": 1, "status": "done", "note": "  файл на месте "},
            verdict="null",
        )
    )

    result = await TaskTracker("gpt-5-nano", 64).observe(
        stub, _executing(), HISTORY, Ledger()
    )

    assert result is not None and result.error is None and result.rejected == []
    assert (result.step.index, result.step.status, result.step.note) == (
        1,
        "done",
        "файл на месте",
    )
    assert result.verdict is None and result.propose is None
    assert result.covered_seq == 2
    role, prompt = (item["content"] for item in stub.requests[0].messages)
    assert "1. Написать файл" in prompt and "Файл написан." in prompt
    # An unfinished step leaves nothing to propose, and the role says so.
    assert "Предлагать смену этапа сейчас нечего" in role


async def test_the_role_names_the_stages_the_machine_would_accept():
    state = _executing()
    done = replace(state, steps=(replace(state.steps[0], status="done"),))
    stub = StubLLM(_tracker_answer())

    await TaskTracker("gpt-5-nano", 64).observe(stub, done, HISTORY, Ledger())

    # The list comes from the machine, so the tracker cannot invent a stage.
    assert "validation (проверка)" in stub.requests[0].messages[0]["content"]


async def test_tracker_receives_the_same_complete_invariant_block():
    stub = StubLLM(_tracker_answer())
    invariants = InvariantSet(
        rules=[
            InvariantRule(
                id="stack.python", category="stack", text="Использовать Python 3.9+."
            )
        ]
    )

    await TaskTracker("gpt-5-nano", 64).observe(
        stub, _executing(), HISTORY, Ledger(), invariants=invariants
    )

    assert [item["role"] for item in stub.requests[0].messages] == [
        "system",
        "system",
        "user",
    ]
    assert stub.requests[0].messages[1]["content"] == invariants.render()


async def test_a_broken_shape_is_refused_without_losing_the_rest():
    stub = StubLLM(
        _tracker_answer(step={"index": "первый"}, verdict=7, propose="validation")
    )

    result = await TaskTracker("gpt-5-nano", 64).observe(
        stub, _executing(), HISTORY, Ledger()
    )

    assert result is not None and result.error is None
    assert result.step is None and result.verdict is None
    assert result.propose == "validation"
    assert result.rejected == ["шаг: индекс 'первый' не число", "вердикт: 7 не строка"]


async def test_an_empty_answer_is_an_error_not_an_empty_decision():
    result = await TaskTracker("gpt-5-nano", 64).observe(
        StubLLM(""), _executing(), HISTORY, Ledger()
    )

    assert result is not None and result.step is None
    assert result.error is not None and "пустой ход трекера" in result.error


async def test_only_the_unread_part_of_the_dialogue_is_sent():
    stub = StubLLM(_tracker_answer())

    result = await TaskTracker("gpt-5-nano", 64).observe(
        stub, _executing(), HISTORY, Ledger(), covered_seq=1
    )

    assert result is not None
    prompt = stub.requests[0].messages[1]["content"]
    assert "Файл написан." in prompt and "Сделай первый шаг." not in prompt
    assert await TaskTracker("gpt-5-nano", 64).observe(
        stub, _executing(), HISTORY, Ledger(), covered_seq=2
    ) is None


# --- the tracker inside a turn ----------------------------------------------


def _task_in_execution(client, **task_spec) -> str:
    """A session whose task is planned, approved and waiting on its first step."""
    session_id = client.post(
        "/api/sessions",
        json={"spec": {"task": {"tracker_enabled": True, **task_spec}}},
    ).json()["id"]
    client.post(f"/api/sessions/{session_id}/task")
    client.app.state.providers["zen"].client = StubLLM(PLAN)
    client.post(f"/api/sessions/{session_id}/messages", json={"text": "Составь план"})
    detail = client.get(f"/api/sessions/{session_id}").json()
    client.post(
        f"/api/sessions/{session_id}/task/transition",
        json={
            "stage": "execution",
            "expected_revision": detail["task"]["revision"],
            "expected_invariant_hash": detail["invariants"]["version_hash"],
        },
    )
    return session_id


def _turn(client, session_id, answer: str, text: str = "Делай") -> list[dict]:
    client.app.state.providers["zen"].client = StubLLM(
        script=[("Сделал.", []), (answer, [])]
    )
    return sse_events(
        client.post(f"/api/sessions/{session_id}/messages", json={"text": text})
    )


def test_the_tracker_closes_the_step_and_only_proposes_the_next_stage(client):
    session_id = _task_in_execution(client)

    events = _turn(
        client,
        session_id,
        _tracker_answer(step={"index": 1, "status": "done"}, propose="validation"),
    )

    marked = next(event for event in events if event["type"] == "task")
    assert marked["task"]["steps"][0]["status"] == "done"
    # A stage that hands out rights is the human's: the tracker only offers it.
    assert marked["proposed"] == "validation"
    assert marked["task"]["stage"] == "execution"
    assert marked["task"]["expected_action"]["actor"] == "user"
    usage = client.get(f"/api/sessions/{session_id}").json()["usage"]
    assert [row["kind"] for row in usage] == ["chat", "chat", "task"]


def test_auto_transitions_let_the_tracker_move_execution_to_validation(client):
    session_id = _task_in_execution(client, auto_transitions=True)

    events = _turn(
        client,
        session_id,
        _tracker_answer(step={"index": 1, "status": "done"}, propose="validation"),
    )

    marked = next(event for event in events if event["type"] == "task")
    assert marked["task"]["stage"] == "validation"
    assert marked["proposed"] is None
    journal = client.get(f"/api/sessions/{session_id}/task").json()["transitions"]
    assert [entry["actor"] for entry in journal][-2:] == ["tracker", "tracker"]


def test_failed_validation_always_waits_for_human_failed_criteria(client):
    session_id = _task_in_execution(client, auto_transitions=True)
    detail = client.get(f"/api/sessions/{session_id}").json()
    client.patch(
        f"/api/sessions/{session_id}/task/steps/1",
        json={"status": "done", "expected_revision": detail["task"]["revision"]},
    )
    detail = client.get(f"/api/sessions/{session_id}").json()
    client.post(
        f"/api/sessions/{session_id}/task/transition",
        json={
            "stage": "validation",
            "expected_revision": detail["task"]["revision"],
            "expected_invariant_hash": detail["invariants"]["version_hash"],
        },
    )

    events = _turn(
        client,
        session_id,
        _tracker_answer(verdict="fail", propose="execution"),
        text="Проверка не прошла",
    )

    marked = next(event for event in events if event["type"] == "task")
    assert marked["task"]["stage"] == "validation"
    assert marked["task"]["verdict"] == "fail"
    assert marked["proposed"] == "execution"


def test_a_step_outside_the_plan_and_a_verdict_outside_validation_are_refused(client):
    session_id = _task_in_execution(client)

    events = _turn(
        client,
        session_id,
        _tracker_answer(step={"index": 9, "status": "done"}, verdict="pass"),
    )

    refused = next(event for event in events if event["type"] == "task_rejected")
    assert refused["reasons"] == [
        "Шага 9 в задаче нет",
        "Вердикт выносится только на этапе проверки",
    ]
    task = client.get(f"/api/sessions/{session_id}").json()["task"]
    assert task["steps"][0]["status"] == "in_progress"
    assert task["verdict"] is None


def test_a_transition_whose_guard_is_unmet_never_leaves_the_tracker(client):
    session_id = _task_in_execution(client, auto_transitions=True)

    events = _turn(client, session_id, _tracker_answer(propose="validation"))

    refused = next(event for event in events if event["type"] == "task_rejected")
    assert refused["reasons"] == ["переход в «проверка» трекеру сейчас недоступен"]
    assert next(event for event in events if event["type"] == "task")["task"][
        "stage"
    ] == "execution"


def test_a_turn_is_read_by_the_tracker_exactly_once(client):
    session_id = _task_in_execution(client)
    covered = client.get(f"/api/sessions/{session_id}").json()["task"]["covered_seq"]

    _turn(client, session_id, _tracker_answer(), text="Первый ход")
    after_first = client.get(f"/api/sessions/{session_id}").json()["task"]
    _turn(client, session_id, _tracker_answer(), text="Второй ход")

    assert after_first["covered_seq"] > covered
    prompt = client.app.state.providers["zen"].client.requests[1].messages[1]["content"]
    assert "Второй ход" in prompt and "Первый ход" not in prompt


def test_the_tracker_skips_the_discussion_that_happened_while_paused(client):
    session_id = _task_in_execution(client)
    task = client.get(f"/api/sessions/{session_id}").json()["task"]
    client.post(
        f"/api/sessions/{session_id}/task/pause",
        json={"expected_revision": task["revision"]},
    )
    client.app.state.providers["zen"].client = StubLLM(
        script=[("120", []), ("среда", [])]
    )
    sse_events(
        client.post(
            f"/api/sessions/{session_id}/messages",
            json={"text": "Сколько минут в двух часах?"},
        )
    )
    sse_events(
        client.post(
            f"/api/sessions/{session_id}/messages",
            json={"text": "Какой день после вторника?"},
        )
    )

    task = client.get(f"/api/sessions/{session_id}").json()["task"]
    client.post(
        f"/api/sessions/{session_id}/task/resume",
        json={"expected_revision": task["revision"]},
    )
    active = StubLLM(
        script=[
            ("Текущий шаг выполнен.", []),
            (_tracker_answer(step={"index": 1, "status": "done"}), []),
        ]
    )
    client.app.state.providers["zen"].client = active
    sse_events(
        client.post(
            f"/api/sessions/{session_id}/messages", json={"text": "Продолжай"}
        )
    )

    prompt = active.requests[1].messages[1]["content"]
    assert "Продолжай" in prompt and "Текущий шаг выполнен" in prompt
    assert "Сколько минут" not in prompt and "Какой день" not in prompt


def test_a_tracker_that_fails_does_not_cancel_the_answer(client):
    session_id = _task_in_execution(client)

    events = _turn(client, session_id, "не json")

    assert events[-1]["type"] == "done"
    failed = next(event for event in events if event["type"] == "task_error")
    assert "не в JSON" in failed["message"]
    # The turn was not read, so the next tracker call sees it again.
    task = client.get(f"/api/sessions/{session_id}").json()["task"]
    assert task["covered_seq"] < max(
        message["seq"]
        for message in client.get(f"/api/sessions/{session_id}").json()["messages"]
    )


def test_a_late_tracker_cannot_advance_a_task_paused_during_its_call(client):
    session_id = _task_in_execution(client)
    snapshot = client.get(f"/api/sessions/{session_id}").json()["task"]
    task = snapshot
    client.post(
        f"/api/sessions/{session_id}/task/pause",
        json={"expected_revision": task["revision"]},
    )
    spec = repo.get_spec(
        client.app.state.db,
        client.app.state.db.execute(
            "SELECT spec_hash FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()["spec_hash"],
    )

    events = _apply_tracker(
        client.app.state.db,
        session_id,
        spec,
        snapshot,
        TrackerResult(
            covered_seq=99,
            step=TrackerStep(index=1, status="done"),
            propose="validation",
        ),
    )

    current = client.get(f"/api/sessions/{session_id}").json()["task"]
    assert current["paused"] is True
    assert current["steps"][0]["status"] == "in_progress"
    assert current["stage"] == "execution"
    assert events[-1]["type"] == "task_rejected"
    assert "не применено" in events[-1]["reasons"][-1]
