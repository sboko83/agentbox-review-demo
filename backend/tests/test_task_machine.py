import pytest

from agentbox.core.task_machine import (
    STAGES,
    TERMINAL_STAGES,
    ParsedPlan,
    Step,
    TaskError,
    TaskState,
    Transition,
    available_transitions,
    expected_action,
    parse_plan,
    permits_tool,
    plan_state,
    render,
    set_paused,
    set_step,
    set_verdict,
    transition,
)

PLAN = "1. Написать a.txt\n2. Написать b.txt\nПроверка:\n- оба файла на месте"
INVARIANT_HASH = "0" * 64

# from_stage -> transitions a human may drive, forcing where a guard asks.
HUMAN_EDGES = {
    "planning": {"planning", "execution", "cancelled"},
    "execution": {"planning", "validation", "cancelled"},
    "validation": {"planning", "execution", "done", "cancelled"},
    "done": set(),
    "cancelled": set(),
}
TRACKER_EDGES = {
    "planning": set(),
    "execution": {"validation"},
    "validation": {"execution"},
    "done": set(),
    "cancelled": set(),
}


def planned() -> TaskState:
    return plan_state(TaskState(), PLAN)


def executing() -> TaskState:
    return transition(planned(), "execution", invariant_hash=INVARIANT_HASH)


def finished() -> TaskState:
    state = executing()
    for step in state.steps:
        state = set_step(state, step.index, status="done")
    return state


def validating(verdict=None) -> TaskState:
    state = transition(finished(), "validation")
    return set_verdict(state, verdict) if verdict else state


def state_at(stage: str, to_stage: str | None = None) -> TaskState:
    # Each stage is built ready for its own transitions, so the table test
    # measures the edges rather than the guards, which have their own tests.
    return {
        "planning": planned,
        "execution": finished,
        "validation": lambda: validating("pass" if to_stage == "done" else "fail"),
        "done": lambda: transition(validating("pass"), "done"),
        "cancelled": lambda: transition(planned(), "cancelled"),
    }[stage]()


@pytest.mark.parametrize("from_stage", STAGES)
@pytest.mark.parametrize("to_stage", STAGES)
@pytest.mark.parametrize("actor", ["user", "tracker"])
def test_every_pair_of_stages_matches_the_transition_table(from_stage, to_stage, actor):
    edges = HUMAN_EDGES if actor == "user" else TRACKER_EDGES
    state = state_at(from_stage, to_stage)

    if to_stage in edges[from_stage]:
        moved = transition(
            state,
            to_stage,
            actor,
            invariant_hash=INVARIANT_HASH,
            failed_criteria=["оба файла на месте"],
        )
        assert moved.stage == to_stage
    else:
        with pytest.raises(TaskError):
            transition(state, to_stage, actor, invariant_hash=INVARIANT_HASH)


def test_execution_needs_a_parsed_plan_with_at_least_one_step():
    draft = plan_state(TaskState(), "сначала разберусь, потом сделаю")

    with pytest.raises(TaskError) as refused:
        transition(draft, "execution")

    assert refused.value.code == "no_steps"
    assert draft.plan_error
    assert transition(planned(), "execution", invariant_hash=INVARIANT_HASH).stage == "execution"


def test_unfinished_steps_stop_validation_even_when_a_human_asks_to_force_it():
    state = executing()

    with pytest.raises(TaskError) as refused:
        transition(state, "validation")
    assert refused.value.code == "unfinished_steps"
    with pytest.raises(TaskError):
        transition(state, "validation", "tracker", force=True)

    with pytest.raises(TaskError) as forced:
        transition(state, "validation", force=True)
    assert forced.value.code == "force_removed"
    assert transition(finished(), "validation", "tracker").stage == "validation"


def test_done_waits_for_a_passing_verdict_and_the_tracker_may_only_propose():
    checking = validating()

    with pytest.raises(TaskError) as refused:
        transition(checking, "done")
    assert refused.value.code == "missing_pass"
    with pytest.raises(TaskError) as by_tracker:
        transition(validating("pass"), "done", "tracker")
    assert by_tracker.value.code == "actor"

    assert transition(validating("pass"), "done").stage == "done"
    with pytest.raises(TaskError) as forced:
        transition(checking, "done", force=True)
    assert forced.value.code == "force_removed"


def test_a_failed_check_turns_unmet_criteria_into_new_steps():
    failed = validating("fail")

    back = transition(
        failed,
        "execution",
        "tracker",
        failed_criteria=["b.txt пуст"],
        invariant_hash=INVARIANT_HASH,
    )

    assert [step.title for step in back.steps][-1] == "b.txt пуст"
    assert [step.index for step in back.steps] == [1, 2, 3]
    assert back.current_step == 3
    assert back.verdict is None


def test_replanning_clears_the_steps_and_keeps_nothing_else():
    state = transition(executing(), "planning")

    assert state.steps == ()
    assert state.plan_text == ""
    assert state.current_step is None
    assert expected_action(state).text == "составить план"


@pytest.mark.parametrize("stage", [s for s in STAGES if s not in TERMINAL_STAGES])
def test_pause_holds_every_non_terminal_stage_and_returns_to_it(stage):
    state = state_at(stage)

    paused = set_paused(state, True)
    assert paused.paused is True
    assert paused.stage == stage
    assert expected_action(paused).text == "продолжить задачу"
    assert available_transitions(paused) == (Transition("cancelled"),)

    with pytest.raises(TaskError) as refused:
        transition(paused, "planning")
    assert refused.value.code == "paused"

    resumed = set_paused(paused, False)
    assert resumed.stage == state.stage
    assert resumed.paused is False
    assert resumed.revision == state.revision + 2


@pytest.mark.parametrize("stage", TERMINAL_STAGES)
def test_a_closed_task_takes_no_pause_and_no_transition(stage):
    state = state_at(stage)

    with pytest.raises(TaskError) as paused:
        set_paused(state, True)
    assert paused.value.code == "terminal"
    assert available_transitions(state) == ()


def test_finishing_a_step_moves_the_pointer_to_the_next_one():
    state = executing()
    assert state.current_step == 1

    state = set_step(state, 1, status="done")

    assert state.current_step == 2
    assert state.step(2).status == "in_progress"
    assert expected_action(state).text == "выполнить шаг 2 «Написать b.txt»"


def test_a_blocked_step_asks_the_human_and_keeps_the_pointer():
    state = set_step(executing(), 1, status="blocked", note="нет прав на запись")

    assert state.current_step == 1
    action = expected_action(state)
    assert action.actor == "user"
    assert "снять блокер шага 1" in action.text
    assert "нет прав на запись" in action.text


def test_the_tracker_does_not_touch_steps_while_planning_or_paused():
    with pytest.raises(TaskError) as planning:
        set_step(planned(), 1, "tracker", status="done")
    assert planning.value.code == "tracker_idle"

    with pytest.raises(TaskError) as paused:
        set_step(set_paused(executing(), True), 1, "tracker", status="done")
    assert paused.value.code == "tracker_idle"


def test_a_step_outside_the_plan_is_refused():
    with pytest.raises(TaskError) as refused:
        set_step(executing(), 7, status="done")
    assert refused.value.code == "step_index"

    with pytest.raises(TaskError) as bad_status:
        set_step(executing(), 1, status="almost")
    assert bad_status.value.code == "step_status"


def test_a_verdict_belongs_to_validation_only():
    with pytest.raises(TaskError) as refused:
        set_verdict(executing(), "pass")

    assert refused.value.code == "verdict_stage"
    assert set_verdict(validating(), "fail").verdict == "fail"


@pytest.mark.parametrize(
    "text, steps, criteria",
    [
        (PLAN, 2, 1),
        ("План:\n1) Первый\n2) Второй\n3) Третий", 3, 0),
        ("1. **Первый**\n2. Второй\n\nКритерии проверки:\n1. тесты зелёные\n2. файл на месте", 2, 2),
        ("- Первый\n- Второй\nПроверка\nответ на русском", 2, 1),
    ],
)
def test_the_parser_tolerates_the_numbering_and_finds_the_criteria(text, steps, criteria):
    parsed = parse_plan(text)

    assert parsed.ok
    assert len(parsed.steps) == steps
    assert len(parsed.criteria) == criteria
    assert "**" not in "".join(parsed.steps)


@pytest.mark.parametrize("text", ["", "Сначала подумаю, потом скажу.", "\n".join(f"{i}. шаг" for i in range(1, 40))])
def test_an_unparsed_plan_stays_a_draft_with_a_reason(text):
    parsed = parse_plan(text)

    assert not parsed.ok
    assert parsed.steps == ()
    assert isinstance(parsed, ParsedPlan)

    draft = plan_state(TaskState(), text)
    assert draft.plan_error == parsed.error
    assert draft.stage == "planning"


def test_a_plan_is_taken_only_while_planning():
    with pytest.raises(TaskError) as refused:
        plan_state(executing(), PLAN)

    assert refused.value.code == "plan_stage"


def test_the_rendered_block_names_the_stage_steps_and_next_action():
    body = render(set_step(executing(), 1, status="done", note="файл создан"))

    assert body.startswith("Этап: выполнение")
    assert "→ 2. Написать b.txt [в работе]" in body
    assert "1. Написать a.txt [готов] — файл создан" in body
    assert "Критерии проверки:" in body
    assert body.endswith("Ожидаемое действие (агент): выполнить шаг 2 «Написать b.txt»")
    assert render(set_paused(executing(), True)).splitlines()[0].endswith("(на паузе)")


def test_the_state_survives_a_round_trip_through_plain_data():
    state = set_step(executing(), 1, status="done", note="готово")

    assert TaskState.from_dict(state.to_dict()) == state
    assert state.to_dict()["expected_action"]["actor"] == "agent"
    assert Step.from_dict({"index": 1, "title": "шаг"}) == Step(1, "шаг")


def test_strict_guards_are_visible_but_never_forceable():
    state = executing()
    validation = next(
        item for item in available_transitions(state) if item.stage == "validation"
    )

    assert validation.allowed is False
    assert validation.needs_force is False
    assert "Не выполнены шаги" in validation.reason
    with pytest.raises(TaskError, match="Принудительные переходы отключены") as refused:
        transition(state, "validation", force=True)
    assert refused.value.code == "force_removed"


def test_changing_the_approved_plan_revokes_execution_rights():
    state = executing()
    assert permits_tool(state, "write_file", True)

    changed = set_step(state, 1, title="Другой первый шаг")

    assert changed.approved_plan_hash is None
    assert not permits_tool(changed, "write_file", True)
    assert expected_action(changed).text.startswith("вернуться в планирование")
    with pytest.raises(TaskError) as refused:
        transition(changed, "validation")
    assert refused.value.code == "stale_approval"


def test_a_result_change_invalidates_the_previous_verdict():
    checked = validating("pass")
    assert checked.verdict_result_revision == checked.result_revision

    changed = set_step(checked, 1, note="результат заменён")

    assert changed.verdict is None
    assert changed.result_revision == checked.result_revision + 1
    with pytest.raises(TaskError) as refused:
        transition(changed, "done")
    assert refused.value.code == "missing_pass"


def test_pause_preserves_approval_result_version_and_verdict():
    checked = validating("pass")

    paused = set_paused(checked, True)
    resumed = set_paused(paused, False)

    assert resumed.approved_plan_hash == checked.approved_plan_hash
    assert resumed.approved_invariant_hash == INVARIANT_HASH
    assert resumed.result_revision == checked.result_revision
    assert resumed.verdict == "pass"
    assert resumed.verdict_result_revision == checked.verdict_result_revision


def test_a_plan_change_after_validation_blocks_done_even_after_a_new_pass():
    changed = set_step(validating("pass"), 1, title="Изменённый шаг")
    checked_again = set_verdict(changed, "pass")

    with pytest.raises(TaskError) as refused:
        transition(checked_again, "done")

    assert refused.value.code == "stale_approval"
    assert "вернуться в планирование" in expected_action(checked_again).text
