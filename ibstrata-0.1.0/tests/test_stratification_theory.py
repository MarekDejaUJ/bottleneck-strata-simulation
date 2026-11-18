"""Numerical checks of Proposition 1 (validity and nesting) and Theorem 1 (information bound)."""

import numpy as np
import pytest
from _helpers import random_joint

from ibstrata.bounds import (
    Event,
    Margins,
    event_probability,
    mix_margins,
    stratified_lp,
    stratified_tian_pearl,
    tian_pearl,
)
from ibstrata.bounds.arms import from_arm_distribution, to_arm_distribution, width_loss_certificate
from ibstrata.info import conditional_mutual_information


def _population(rng, nc, m, n):
    joints = [random_joint(rng, m, n, float(rng.choice([0.3, 1.0, 3.0]))) for _ in range(nc)]
    margins = [Margins.from_joint(j) for j in joints]
    p_c = rng.dirichlet(np.ones(nc))
    pi = rng.dirichlet(np.full(m + 1, 4.0))
    return joints, margins, p_c, pi


def _strata(margins, p_c, pi, enc):
    m, n = margins[0].m, margins[0].n
    q_c = np.array([to_arm_distribution(mg, pi) for mg in margins])
    p_ct = p_c[:, None] * enc
    p_t = p_ct.sum(axis=0)
    keep = p_t > 1e-12
    q_t = (p_ct[:, keep] / p_t[keep]).T @ q_c
    margins_t = [from_arm_distribution(q, m, n, pi) for q in q_t]
    return q_c, margins_t, p_t[keep]


def test_arm_distribution_round_trip(rng):
    _, margins, _, pi = _population(rng, 1, 3, 2)
    q = to_arm_distribution(margins[0], pi)
    assert q.sum() == pytest.approx(1.0)
    back = from_arm_distribution(q, 3, 2, pi)
    np.testing.assert_allclose(back.experimental, margins[0].experimental)
    np.testing.assert_allclose(back.observational, margins[0].observational)


def test_binary_nesting_validity_and_information_bound(rng):
    worst_ratio = 0.0
    for _ in range(300):
        nc, k = int(rng.integers(2, 9)), int(rng.integers(1, 4))
        joints, margins, p_c, pi = _population(rng, nc, 2, 2)
        enc = rng.dirichlet(np.full(k, float(rng.choice([0.05, 0.5, 5.0]))), size=nc)
        q_c, margins_t, p_t = _strata(margins, p_c, pi, enc)
        by_c = stratified_tian_pearl(margins, p_c)
        by_t = stratified_tian_pearl(margins_t, p_t)
        pooled = tian_pearl(mix_margins(margins, p_c))
        truth = sum(pc * event_probability(j, Event(do={1: 1, 0: 0})) for pc, j in zip(p_c, joints))
        tol = 1e-9
        assert pooled.pns.lower <= by_t.pns.lower + tol <= by_c.pns.lower + 2 * tol
        assert by_c.pns.upper <= by_t.pns.upper + tol <= pooled.pns.upper + 2 * tol
        assert by_c.pns.contains(truth)
        loss = (by_c.pns.lower - by_t.pns.lower) + (by_t.pns.upper - by_c.pns.upper)
        info = conditional_mutual_information(p_c, q_c, enc)
        certificate = width_loss_certificate(info, pi)
        assert loss <= certificate + 1e-9
        if certificate > 1e-6:
            worst_ratio = max(worst_ratio, loss / certificate)
    assert worst_ratio <= 1.0


def test_exact_lp_nesting_for_multivalued_outcome(rng):
    event = Event(do={0: 2, 1: 0})
    for _ in range(40):
        nc, k = int(rng.integers(2, 6)), int(rng.integers(1, 3))
        joints, margins, p_c, pi = _population(rng, nc, 2, 3)
        enc = rng.dirichlet(np.full(k, 0.5), size=nc)
        _, margins_t, p_t = _strata(margins, p_c, pi, enc)
        by_c = stratified_lp(margins, p_c, event)
        by_t = stratified_lp(margins_t, p_t, event)
        truth = sum(pc * event_probability(j, event) for pc, j in zip(p_c, joints))
        assert by_t.lower <= by_c.lower + 1e-8 and by_c.upper <= by_t.upper + 1e-8
        assert by_c.contains(truth, tol=1e-8)
