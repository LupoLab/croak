"""The window's route through a session, as rules without Qt.

A session runs Home → Data → Prepare → Retrieve → Analyse. Which stages can be
opened, and what the one primary action in the top bar does, depend only on what
the session holds (:class:`Progress`), so they are decided here, as pure
functions the window applies and the tests check directly.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

__all__ = [
    "STAGES",
    "Action",
    "Progress",
    "Stage",
    "blocked_reason",
    "next_stage",
    "primary_action",
    "remember_recent",
]

RECENT_LIMIT = 6
"""How many recent sessions Home lists."""


class Stage(StrEnum):
    """A place in the window."""

    HOME = "home"
    DATA = "data"
    PREPARE = "prepare"
    RETRIEVE = "retrieve"
    ANALYSE = "analyse"

    @property
    def title(self) -> str:
        """The stage's name in the top bar."""
        return self.value.capitalize()


STAGES = (Stage.DATA, Stage.PREPARE, Stage.RETRIEVE, Stage.ANALYSE)
"""The stages of a session, in order (Home is outside the session)."""


@dataclass(frozen=True)
class Progress:
    """What the current session holds, as far as the route is concerned.

    Attributes
    ----------
    entry : str
        ``"experimental"``, ``"simulated"`` or ``"synthetic"``; ``""`` before a
        session starts.
    data_ready : bool
        The Data sheet can hand on a trace (a file previews, a synthetic pulse
        is set up).
    has_load_data : bool
        A trace has been loaded (``state.load_data``).
    has_tracedata : bool
        A retrieval trace exists (``state.tracedata``).
    has_result : bool
        A retrieval result exists.
    raw_direct : bool
        The simulated scan goes to Retrieve on its native grid, skipping Prepare.
    retrieving : bool
        A retrieval is running.
    estimating : bool
        An uncertainty estimate is running.
    """

    entry: str = ""
    data_ready: bool = False
    has_load_data: bool = False
    has_tracedata: bool = False
    has_result: bool = False
    raw_direct: bool = False
    retrieving: bool = False
    estimating: bool = False


def blocked_reason(stage: Stage, progress: Progress) -> str:
    """Why ``stage`` cannot be opened now, or ``""`` if it can.

    Examples
    --------
    >>> blocked_reason(Stage.PREPARE, Progress(entry="experimental"))
    'Load a trace first.'
    >>> blocked_reason(Stage.DATA, Progress(entry="synthetic"))
    ''
    """
    if stage is Stage.HOME:
        return ""
    if not progress.entry:
        return "Start a session from Home first."
    if stage is Stage.DATA:
        return ""
    if stage is Stage.PREPARE:
        if progress.raw_direct:
            return (
                "This simulated scan is retrieved on its own grid "
                "(Skip filtering & regrid), so there is nothing to prepare."
            )
        return "" if progress.has_load_data else "Load a trace first."
    if stage is Stage.RETRIEVE:
        return "" if progress.has_tracedata else "Prepare the trace first."
    return "" if progress.has_result else "Run a retrieval first."


def next_stage(stage: Stage, progress: Progress) -> Stage:
    """The stage "Continue" leads to from ``stage``.

    From Data a raw simulated scan goes straight to Retrieve.

    Raises
    ------
    ValueError
        For a stage with no "Continue" (Home, Retrieve, Analyse).
    """
    if stage is Stage.DATA:
        return Stage.RETRIEVE if progress.raw_direct else Stage.PREPARE
    if stage is Stage.PREPARE:
        return Stage.RETRIEVE
    raise ValueError(f"{stage.title} has no next stage")


@dataclass(frozen=True)
class Action:
    """The top bar's primary action.

    Attributes
    ----------
    kind : str
        What it does: ``"continue"``, ``"retrieve"``, ``"stop"``, ``"save"``,
        ``"estimate"`` or ``"stop-estimate"``; ``""`` for no action.
    text : str
        The button's caption.
    enabled : bool
        Whether it can be pressed now.
    """

    kind: str
    text: str
    enabled: bool = True


def primary_action(stage: Stage, progress: Progress, page: str = "") -> Action:
    """The primary action on ``stage`` (and Analyse ``page``) given ``progress``.

    Examples
    --------
    >>> primary_action(Stage.RETRIEVE, Progress(entry="synthetic", retrieving=True))
    Action(kind='stop', text='Stop', enabled=True)
    >>> primary_action(Stage.DATA, Progress(entry="synthetic")).enabled
    False
    """
    if stage is Stage.HOME:
        return Action("", "", enabled=False)
    if stage is Stage.DATA:
        return Action("continue", "Continue", enabled=progress.data_ready)
    if stage is Stage.PREPARE:
        return Action("continue", "Continue", enabled=progress.has_tracedata)
    if stage is Stage.RETRIEVE:
        if progress.retrieving:
            return Action("stop", "Stop")
        text = "Retrieve again" if progress.has_result else "Retrieve"
        return Action("retrieve", text, enabled=progress.has_tracedata)
    if page == "uncertainty":
        if progress.estimating:
            return Action("stop-estimate", "Stop")
        return Action("estimate", "Estimate", enabled=progress.has_result)
    return Action("save", "Save", enabled=progress.has_result)


def remember_recent(
    recent: Sequence[str], path: str, limit: int = RECENT_LIMIT
) -> list[str]:
    """Put ``path`` first in the recent list, without repeats, at most ``limit``.

    Examples
    --------
    >>> remember_recent(["a", "b", "c"], "b", limit=3)
    ['b', 'a', 'c']
    >>> remember_recent(["a", "b"], "z", limit=2)
    ['z', 'a']
    """
    if limit < 1:
        raise ValueError(f"limit must be at least 1, not {limit}")
    return [path, *(p for p in recent if p != path)][:limit]
