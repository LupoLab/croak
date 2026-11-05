"""The window's route rules: which stages open and what the primary action does."""

from __future__ import annotations

import pytest

from croak.gui.workflow import (
    STAGES,
    Action,
    Progress,
    Stage,
    blocked_reason,
    next_stage,
    primary_action,
    remember_recent,
)

FRESH = Progress(entry="experimental")
LOADED = Progress(entry="experimental", data_ready=True, has_load_data=True)
PREPARED = Progress(
    entry="experimental", data_ready=True, has_load_data=True, has_tracedata=True
)
RETRIEVED = Progress(
    entry="experimental",
    data_ready=True,
    has_load_data=True,
    has_tracedata=True,
    has_result=True,
)


def test_workflow_stages_are_in_session_order_with_titles():
    assert [s.title for s in STAGES] == ["Data", "Prepare", "Retrieve", "Analyse"]
    assert Stage.HOME not in STAGES


@pytest.mark.parametrize(
    ("stage", "progress", "blocked"),
    [
        (Stage.HOME, Progress(), False),
        (Stage.DATA, Progress(), True),  # no session yet
        (Stage.DATA, FRESH, False),
        (Stage.PREPARE, FRESH, True),
        (Stage.PREPARE, LOADED, False),
        (Stage.RETRIEVE, LOADED, True),
        (Stage.RETRIEVE, PREPARED, False),
        (Stage.ANALYSE, PREPARED, True),
        (Stage.ANALYSE, RETRIEVED, False),
    ],
)
def test_workflow_stage_opens_only_once_its_input_exists(stage, progress, blocked):
    assert bool(blocked_reason(stage, progress)) is blocked


def test_workflow_raw_simulated_scan_skips_prepare_with_a_reason():
    raw = Progress(entry="simulated", has_load_data=True, raw_direct=True)
    assert "Skip filtering & regrid" in blocked_reason(Stage.PREPARE, raw)
    assert next_stage(Stage.DATA, raw) is Stage.RETRIEVE


@pytest.mark.parametrize(
    ("stage", "expected"),
    [(Stage.DATA, Stage.PREPARE), (Stage.PREPARE, Stage.RETRIEVE)],
)
def test_workflow_next_stage_follows_the_session(stage, expected):
    assert next_stage(stage, LOADED) is expected


@pytest.mark.parametrize("stage", [Stage.HOME, Stage.RETRIEVE, Stage.ANALYSE])
def test_workflow_next_stage_refuses_a_stage_without_continue(stage):
    with pytest.raises(ValueError, match="has no next stage"):
        next_stage(stage, LOADED)


@pytest.mark.parametrize(
    ("stage", "progress", "page", "expected"),
    [
        (Stage.HOME, FRESH, "", Action("", "", enabled=False)),
        (Stage.DATA, FRESH, "", Action("continue", "Continue", enabled=False)),
        (Stage.DATA, LOADED, "", Action("continue", "Continue")),
        (Stage.PREPARE, LOADED, "", Action("continue", "Continue", enabled=False)),
        (Stage.PREPARE, PREPARED, "", Action("continue", "Continue")),
        (Stage.RETRIEVE, LOADED, "", Action("retrieve", "Retrieve", enabled=False)),
        (Stage.RETRIEVE, PREPARED, "", Action("retrieve", "Retrieve")),
        (Stage.RETRIEVE, RETRIEVED, "", Action("retrieve", "Retrieve again")),
        (Stage.ANALYSE, RETRIEVED, "overview", Action("save", "Save")),
        (Stage.ANALYSE, PREPARED, "", Action("save", "Save", enabled=False)),
        (Stage.ANALYSE, RETRIEVED, "uncertainty", Action("estimate", "Estimate")),
    ],
)
def test_workflow_primary_action_is_what_comes_next(stage, progress, page, expected):
    assert primary_action(stage, progress, page) == expected


def test_workflow_primary_action_stops_running_work():
    running = Progress(entry="synthetic", has_tracedata=True, retrieving=True)
    assert primary_action(Stage.RETRIEVE, running).kind == "stop"
    estimating = Progress(entry="synthetic", has_result=True, estimating=True)
    action = primary_action(Stage.ANALYSE, estimating, "uncertainty")
    assert action == Action("stop-estimate", "Stop")


def test_workflow_recent_list_moves_a_path_first_and_caps_its_length():
    assert remember_recent([], "a") == ["a"]
    assert remember_recent(["a", "b", "c"], "c", limit=2) == ["c", "a"]
    with pytest.raises(ValueError, match="at least 1"):
        remember_recent(["a"], "b", limit=0)
