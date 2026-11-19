"""The arm-augmented outcome V = (S, X, Y) of a design with forced arms and a free-choice arm.

Layout of a distribution over V for ``m`` treatment values and ``n`` outcome values:
first ``m`` blocks of ``n`` cells, block ``j`` holding ``pi_j * P(Y = y | S = forced x_j)``;
then one block of ``m * n`` cells holding ``pi_o * P(X = x, Y = y | S = choice)`` in
row-major order. Experimental and observational margins are linear in this vector,
which is what makes the stratified bounds convex or concave in it.
"""

from __future__ import annotations

import numpy as np

from .margins import Margins


def check_allocation(pi: np.ndarray, m: int) -> np.ndarray:
    pi = np.asarray(pi, dtype=float)
    if pi.shape != (m + 1,) or (pi <= 0).any() or not np.isclose(pi.sum(), 1.0):
        raise ValueError("allocation needs m + 1 positive probabilities summing to one")
    return pi


def to_arm_distribution(margins: Margins, pi: np.ndarray) -> np.ndarray:
    """Distribution of V implied by margins and arm probabilities ``pi`` (forced arms
    in treatment order, then the choice arm)."""
    m = margins.m
    pi = check_allocation(pi, m)
    forced = (pi[:m, None] * margins.experimental).ravel()
    choice = pi[m] * margins.observational.ravel()
    return np.concatenate([forced, choice])


def from_arm_distribution(q: np.ndarray, m: int, n: int, pi: np.ndarray) -> Margins:
    """Margins identified by a distribution of V."""
    pi = check_allocation(pi, m)
    q = np.asarray(q, dtype=float)
    if q.shape != (2 * m * n,):
        raise ValueError("q has the wrong length for (m, n)")
    exp = q[: m * n].reshape(m, n) / pi[:m, None]
    obs = q[m * n :].reshape(m, n) / pi[m]
    return Margins(exp, obs)


def arm_index(m: int, n: int) -> np.ndarray:
    """Arm label of every cell of V: 0..m-1 for forced arms, m for the choice arm."""
    return np.concatenate([np.repeat(np.arange(m), n), np.full(m * n, m)])


def tightness_constant(pi: np.ndarray, a: np.ndarray | float = 1.0,
                       b: np.ndarray | float = 1.0) -> float:
    """kappa = sqrt(sum_s (a_s + b_s)^2 / pi_s) of Theorem 1.

    ``a`` and ``b`` are the per-arm total-variation coefficients of the lower and upper
    bound functionals; one per arm each for the Tian-Pearl PNS bounds.
    """
    pi = np.asarray(pi, dtype=float)
    a = np.broadcast_to(np.asarray(a, dtype=float), pi.shape)
    b = np.broadcast_to(np.asarray(b, dtype=float), pi.shape)
    return float(np.sqrt(np.sum((a + b) ** 2 / pi)))


def width_loss_certificate(conditional_information: float, pi: np.ndarray,
                           a: np.ndarray | float = 1.0, b: np.ndarray | float = 1.0) -> float:
    """Theorem 1 bound on the width lost by stratifying on T instead of C:
    kappa * sqrt(I(C; V | T) / 2), with the information in nats."""
    return tightness_constant(pi, a, b) * float(np.sqrt(max(conditional_information, 0.0) / 2.0))
