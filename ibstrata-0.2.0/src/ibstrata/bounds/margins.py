"""Experimental and observational margins of a treatment-outcome pair."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import product

import numpy as np

_TOL = 1e-9


@dataclass(frozen=True)
class Margins:
    """Margins combined from experimental and observational data.

    Parameters
    ----------
    experimental
        Array of shape ``(m, n)``. Row ``j`` is the distribution of the potential
        outcome ``Y_{x_j}`` over the ``n`` outcome values, as identified by a
        randomized arm that sets ``X = x_j``.
    observational
        Array of shape ``(m, n)`` holding the joint distribution ``P(X = x_j, Y = y_i)``
        in the population whose units choose their treatment.
    x_labels, y_labels
        Optional labels of the treatment and outcome values.
    """

    experimental: np.ndarray
    observational: np.ndarray
    x_labels: tuple[str, ...] = field(default=())
    y_labels: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        exp = np.asarray(self.experimental, dtype=float)
        obs = np.asarray(self.observational, dtype=float)
        if exp.ndim != 2 or obs.ndim != 2 or exp.shape != obs.shape:
            raise ValueError("experimental and observational must be arrays of equal shape (m, n)")
        m, n = exp.shape
        if m < 2 or n < 2:
            raise ValueError("treatment and outcome need at least two values each")
        if (exp < -_TOL).any() or (obs < -_TOL).any():
            raise ValueError("probabilities must be nonnegative")
        if not np.allclose(exp.sum(axis=1), 1.0, atol=1e-7):
            raise ValueError("each row of the experimental array must sum to one")
        if not np.isclose(obs.sum(), 1.0, atol=1e-7):
            raise ValueError("the observational array must sum to one")
        object.__setattr__(self, "experimental", np.clip(exp, 0.0, None))
        object.__setattr__(self, "observational", np.clip(obs, 0.0, None))
        xl = tuple(self.x_labels) or tuple(f"x{j + 1}" for j in range(m))
        yl = tuple(self.y_labels) or tuple(f"y{i + 1}" for i in range(n))
        if len(xl) != m or len(yl) != n:
            raise ValueError("label lengths must match the array shape")
        object.__setattr__(self, "x_labels", xl)
        object.__setattr__(self, "y_labels", yl)

    @property
    def m(self) -> int:
        """Number of treatment values."""
        return self.experimental.shape[0]

    @property
    def n(self) -> int:
        """Number of outcome values."""
        return self.experimental.shape[1]

    @property
    def p_x(self) -> np.ndarray:
        """Observational distribution of the chosen treatment."""
        return self.observational.sum(axis=1)

    @property
    def p_y(self) -> np.ndarray:
        """Observational distribution of the outcome."""
        return self.observational.sum(axis=0)

    def compatibility_gap(self) -> float:
        """Smallest value of ``P(Y_x = y) - P(X = x, Y = y)`` over all cells.

        Experimental and observational margins come from one population exactly when
        every cell is nonnegative: consistency forces ``P(Y_x = y) >= P(x, y)``, and when
        the inequality holds a joint distribution of the choice and all potential
        outcomes reproduces both margins.
        """
        return float((self.experimental - self.observational).min())

    def is_compatible(self, tol: float = 1e-9) -> bool:
        """True when the two kinds of margins can come from one population."""
        return self.compatibility_gap() >= -tol

    @classmethod
    def from_counts(
        cls,
        experimental_counts: Sequence[Sequence[float]],
        observational_counts: Sequence[Sequence[float]],
        **labels: tuple[str, ...],
    ) -> Margins:
        """Build margins from counts: one row of outcome counts per forced arm, and the
        cross-table of chosen treatment by outcome in the free-choice arm."""
        e = np.asarray(experimental_counts, dtype=float)
        o = np.asarray(observational_counts, dtype=float)
        if (e.sum(axis=1) <= 0).any() or o.sum() <= 0:
            raise ValueError("every arm needs at least one unit")
        return cls(e / e.sum(axis=1, keepdims=True), o / o.sum(), **labels)

    @classmethod
    def from_joint(cls, joint: np.ndarray) -> Margins:
        """Margins implied by a joint distribution of the choice and all potential outcomes.

        ``joint`` has shape ``(m, n, ..., n)`` with ``m`` outcome axes: entry
        ``joint[x, y_1, ..., y_m]`` is ``P(X = x, Y_{x_1} = y_1, ..., Y_{x_m} = y_m)``.
        """
        joint = np.asarray(joint, dtype=float)
        m = joint.shape[0]
        if joint.ndim != m + 1:
            raise ValueError("joint must have one choice axis and one axis per treatment value")
        n = joint.shape[1]
        exp = np.zeros((m, n))
        obs = np.zeros((m, n))
        for x in range(m):
            other = tuple(a for a in range(m + 1) if a != x + 1)
            exp[x] = joint.sum(axis=other)
            obs[x] = joint[x].sum(axis=tuple(a for a in range(m) if a != x))
        return cls(exp, obs)


def response_cells(m: int, n: int) -> list[tuple[int, tuple[int, ...]]]:
    """All cells ``(x, (y_1, ..., y_m))`` of the joint of the choice and the potential outcomes."""
    return [(x, ys) for x in range(m) for ys in product(range(n), repeat=m)]


def project_compatible(margins: Margins, n_experimental: np.ndarray | float = 1.0,
                       n_observational: float = 1.0) -> tuple[Margins, float]:
    """The compatible margins closest to ``margins`` in weighted l1 distance.

    Sampling noise can produce estimated margins with ``P(Y_x = y) < P(X = x, Y = y)`` in
    some cell, which no population can generate. The projection solves a linear program
    over experimental rows and an observational table that satisfy every compatibility
    inequality, minimizing the sum of absolute changes weighted by the sample size of each
    arm (``n_experimental`` per forced arm, ``n_observational`` for the choice arm).
    Returns the projected margins and the weighted l1 distance moved (zero when the input
    is compatible).
    """
    from scipy.optimize import linprog

    if margins.is_compatible(tol=0.0):
        return margins, 0.0
    m, n = margins.m, margins.n
    k = m * n
    we = np.broadcast_to(np.asarray(n_experimental, dtype=float), (m,))
    w = np.concatenate([np.repeat(we, n), np.repeat(we, n),
                        np.full(k, float(n_observational)), np.full(k, float(n_observational))])
    nv = 6 * k  # e, o, de_plus, de_minus, do_plus, do_minus
    c = np.concatenate([np.zeros(2 * k), w])
    a_eq, b_eq = [], []
    e_hat = margins.experimental.ravel()
    o_hat = margins.observational.ravel()
    for i in range(k):
        row = np.zeros(nv)
        row[i], row[2 * k + i], row[3 * k + i] = 1, -1, 1
        a_eq.append(row)
        b_eq.append(e_hat[i])
    for i in range(k):
        row = np.zeros(nv)
        row[k + i], row[4 * k + i], row[5 * k + i] = 1, -1, 1
        a_eq.append(row)
        b_eq.append(o_hat[i])
    for j in range(m):
        row = np.zeros(nv)
        row[j * n:(j + 1) * n] = 1
        a_eq.append(row)
        b_eq.append(1.0)
    row = np.zeros(nv)
    row[k:2 * k] = 1
    a_eq.append(row)
    b_eq.append(1.0)
    a_ub, b_ub = [], []
    for i in range(k):
        row = np.zeros(nv)
        row[k + i], row[i] = 1, -1
        a_ub.append(row)
        b_ub.append(0.0)
    res = linprog(c, A_ub=np.array(a_ub), b_ub=np.array(b_ub), A_eq=np.array(a_eq),
                  b_eq=np.array(b_eq), bounds=(0, None), method="highs")
    if res.status != 0:
        raise ValueError(f"projection failed: {res.message}")
    e = res.x[:k].reshape(m, n)
    o = res.x[k:2 * k].reshape(m, n)
    return Margins(e, o, margins.x_labels, margins.y_labels), float(res.fun)
