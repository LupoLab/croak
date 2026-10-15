"""A scripted stand-in for ``nlopt.opt`` to exercise the solvers' NLopt contracts.

Real NLopt runs only rarely hit two paths the solvers must handle: a value-only
objective call (an empty gradient buffer, which NLopt uses whenever it does not
need derivatives) and an internal line-search failure (``RoundoffLimited`` /
``runtime_error``) that the solvers salvage. :func:`script_nlopt` replaces
``nlopt.opt`` for one test so both can be triggered deterministically.
"""

from __future__ import annotations

import nlopt
import numpy as np


class ScriptedOpt:
    """Wrap a real ``nlopt.opt``, scripting what ``optimize`` does.

    ``value_only_probe`` makes ``optimize`` first call the objective with an
    empty gradient buffer, then run the real optimiser. ``fail_after`` makes it
    instead evaluate the objective (with gradient) that many times at ``u0`` and
    then raise ``error``, as NLopt does when its line search gives up.
    """

    def __init__(self, real, *, value_only_probe=False, fail_after=None, error=None):
        self._real = real
        self._objective = None
        self.value_only_probe = value_only_probe
        self.fail_after = fail_after
        self.error = error or nlopt.RoundoffLimited()

    def __getattr__(self, name):
        return getattr(self._real, name)

    def set_min_objective(self, f):
        self._objective = f
        self._real.set_min_objective(f)

    def optimize(self, u0):
        u0 = np.asarray(u0, dtype=float)
        if self.value_only_probe:
            self._objective(u0, np.empty(0))
        if self.fail_after is not None:
            for _ in range(self.fail_after):
                self._objective(u0, np.empty(u0.size))
            raise self.error
        return self._real.optimize(u0)


def script_nlopt(monkeypatch, **kwargs) -> list[ScriptedOpt]:
    """Make every ``nlopt.opt(...)`` in this test a :class:`ScriptedOpt`.

    Returns the list the created optimisers are appended to.
    """
    created: list[ScriptedOpt] = []
    real_opt = nlopt.opt

    def factory(*args):
        opt = ScriptedOpt(real_opt(*args), **kwargs)
        created.append(opt)
        return opt

    monkeypatch.setattr(nlopt, "opt", factory)
    return created
