"""Exact bounds by linear programming over the joint of the choice and the potential outcomes.

The unknown is the distribution ``z`` of ``(X, Y_{x_1}, ..., Y_{x_m})``, with ``m * n**m``
cells. Consistency ties the observed outcome to the potential outcome of the chosen
treatment, so the observational margin fixes ``P(X = x_j, Y_{x_j} = y_i)`` and each
randomized arm fixes the margin of ``Y_{x_j}``. The minimum and the maximum of the
probability of an event over all ``z`` that reproduce both margins are the sharp
bounds of the event given those margins (Balke's construction, used by Tian and Pearl,
2000).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from .events import Event
from .margins import Margins, response_cells
from .result import Bounds

_CACHE: dict[tuple[int, int], tuple[list, object, list[tuple[str, int, int]]]] = {}


def _system(m: int, n: int):
    key = (m, n)
    if key not in _CACHE:
        cells = response_cells(m, n)
        rows, cols = [], []
        labels: list[tuple[str, int, int]] = []
        r = 0
        for x in range(m):
            for y in range(n):
                labels.append(("obs", x, y))
                for c, (cx, ys) in enumerate(cells):
                    if cx == x and ys[x] == y:
                        rows.append(r)
                        cols.append(c)
                r += 1
        for x in range(m):
            for y in range(n):
                labels.append(("exp", x, y))
                for c, (_, ys) in enumerate(cells):
                    if ys[x] == y:
                        rows.append(r)
                        cols.append(c)
                r += 1
        a = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(r, len(cells))).tocsr()
        _CACHE[key] = (cells, a, labels)
    return _CACHE[key]


def _rhs(margins: Margins, labels) -> np.ndarray:
    return np.array(
        [margins.observational[x, y] if kind == "obs" else margins.experimental[x, y]
         for kind, x, y in labels]
    )


def lp_bounds(margins: Margins, event: Event, given: Event | None = None) -> Bounds:
    """Sharp bounds on ``P(event)`` or on ``P(event | given)``.

    ``given`` may only involve observed variables (``x`` and ``y``); the conditional
    bounds are the joint bounds divided by the observational probability of ``given``.

    Raises
    ------
    ValueError
        When the margins are incompatible (no joint distribution reproduces them) or
        the conditioning event has probability zero.
    """
    m, n = margins.m, margins.n
    if not margins.is_compatible(tol=1e-9):
        raise ValueError(
            "incompatible margins: some P(Y_x = y) is below P(X = x, Y = y) "
            f"(gap {margins.compatibility_gap():.3g})"
        )
    target = event
    denom = 1.0
    if given is not None:
        if not given.observed_only():
            raise ValueError("the conditioning event may involve observed variables only")
        target = event.conjoin(given)
        denom = _observed_probability(margins, given)
        if denom <= 0:
            raise ValueError("the conditioning event has probability zero")
    cells, a, labels = _system(m, n)
    b = _rhs(margins, labels)
    c = np.array([1.0 if target.holds(x, ys) else 0.0 for x, ys in cells])
    lo = linprog(c, A_eq=a, b_eq=b, bounds=(0, None), method="highs")
    hi = linprog(-c, A_eq=a, b_eq=b, bounds=(0, None), method="highs")
    if lo.status != 0 or hi.status != 0:
        raise ValueError(f"linear program failed: {lo.message} / {hi.message}")
    lower = max(0.0, float(lo.fun)) / denom
    upper = min(1.0 * denom, float(-hi.fun)) / denom
    return Bounds(lower=lower, upper=min(upper, 1.0), method="exact LP",
                  lower_witness=np.asarray(lo.x), upper_witness=np.asarray(hi.x))


def lp_value_dual_norm(margins: Margins, event: Event) -> tuple[float, float]:
    """Largest absolute dual value at the optimum of the lower and the upper program.

    The optimal value of a linear program changes by at most the largest absolute
    dual value times the l1 change of the right-hand side, which gives a local
    Lipschitz constant of the exact bounds in the margins.
    """
    cells, a, labels = _system(margins.m, margins.n)
    b = _rhs(margins, labels)
    c = np.array([1.0 if event.holds(x, ys) else 0.0 for x, ys in cells])
    lo = linprog(c, A_eq=a, b_eq=b, bounds=(0, None), method="highs")
    hi = linprog(-c, A_eq=a, b_eq=b, bounds=(0, None), method="highs")
    return float(np.abs(lo.eqlin.marginals).max()), float(np.abs(hi.eqlin.marginals).max())


def _observed_probability(margins: Margins, event: Event) -> float:
    obs = margins.observational
    if event.x is not None and event.y is not None:
        return float(obs[event.x, event.y])
    if event.x is not None:
        return float(obs[event.x].sum())
    if event.y is not None:
        return float(obs[:, event.y].sum())
    return 1.0


def event_probability(joint: np.ndarray, event: Event) -> float:
    """Probability of an event under a joint distribution of shape ``(m, n, ..., n)``."""
    joint = np.asarray(joint, dtype=float)
    m = joint.shape[0]
    n = joint.shape[1]
    total = 0.0
    for x, ys in response_cells(m, n):
        if event.holds(x, ys):
            total += joint[(x, *ys)]
    return total
