"""Numerical checks of Proposition 1 (validity and nesting), Proposition 2 (width lost by a
stratifier), Proposition 3 (narrowest bounds of k strata) and Theorem 1 (information bound)."""

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


def _pns_term_values(mg):
    from ibstrata.stratify import pns_terms
    a_lo, a_hi = pns_terms()
    p = np.concatenate([mg.experimental.ravel(), mg.observational.ravel()])
    return a_lo @ p, a_hi @ p


def test_pns_terms_reproduce_the_tian_pearl_bounds(rng):
    for _ in range(200):
        _, margins, _, _ = _population(rng, 1, 2, 2)
        lo, hi = _pns_term_values(margins[0])
        b = tian_pearl(margins[0]).pns
        assert lo.max() == pytest.approx(b.lower, abs=1e-12)
        assert hi.min() == pytest.approx(b.upper, abs=1e-12)


def test_exact_loss_identity(rng):
    """The width lost by a stratifier equals the expected gap, within the strata, between
    the binding term at a covariate value and the term that binds at the stratum."""
    for _ in range(300):
        nc, k = int(rng.integers(2, 9)), int(rng.integers(1, 4))
        _, margins, p_c, pi = _population(rng, nc, 2, 2)
        enc = rng.dirichlet(np.full(k, float(rng.choice([0.05, 0.5, 5.0]))), size=nc)
        _, margins_t, p_t = _strata(margins, p_c, pi, enc)
        by_c = stratified_tian_pearl(margins, p_c)
        by_t = stratified_tian_pearl(margins_t, p_t)
        terms_c = [_pns_term_values(mg) for mg in margins]
        p_ct = (p_c[:, None] * enc)[:, (p_c[:, None] * enc).sum(axis=0) > 1e-12]
        gap_lo = gap_hi = 0.0
        for t, mg in enumerate(margins_t):
            lo_t, hi_t = _pns_term_values(mg)
            i, j = int(lo_t.argmax()), int(hi_t.argmin())
            for c in range(nc):
                gap_lo += p_ct[c, t] * (terms_c[c][0].max() - terms_c[c][0][i])
                gap_hi += p_ct[c, t] * (terms_c[c][1][j] - terms_c[c][1].min())
        assert by_c.pns.lower - by_t.pns.lower == pytest.approx(gap_lo, abs=1e-10)
        assert by_t.pns.upper - by_c.pns.upper == pytest.approx(gap_hi, abs=1e-10)


def test_binding_pattern_partition_attains_the_full_covariate_bounds(rng):
    """Strata formed by the pair of binding terms at every covariate value give the
    bounds of the full covariates with at most sixteen strata."""
    for _ in range(100):
        nc = int(rng.integers(5, 60))
        _, margins, p_c, pi = _population(rng, nc, 2, 2)
        label = np.array([int(lo.argmax()) * 4 + int(hi.argmin())
                          for lo, hi in map(_pns_term_values, margins)])
        codes = np.unique(label, return_inverse=True)[1]
        enc = np.eye(codes.max() + 1)[codes]
        _, margins_t, p_t = _strata(margins, p_c, pi, enc)
        by_c = stratified_tian_pearl(margins, p_c)
        by_t = stratified_tian_pearl(margins_t, p_t)
        assert len(margins_t) <= 16
        assert by_t.pns.lower == pytest.approx(by_c.pns.lower, abs=1e-10)
        assert by_t.pns.upper == pytest.approx(by_c.pns.upper, abs=1e-10)


def test_narrowest_k_strata_are_strata_of_term_pairs(rng):
    """No partition into k strata is narrower than the partition by the k best pairs of
    terms, and that partition attains the mean of the narrowest chosen pair width."""
    from itertools import product

    from ibstrata.stratify import best_term_pairs, pair_widths
    for _ in range(40):
        nc, k = int(rng.integers(4, 8)), int(rng.integers(1, 4))
        _, margins, p_c, pi = _population(rng, nc, 2, 2)
        q_c = np.array([to_arm_distribution(mg, pi) for mg in margins])
        w = pair_widths(q_c, pi)
        full = stratified_tian_pearl(margins, p_c).pns
        np.testing.assert_allclose(w.min(axis=1) @ p_c, full.upper - full.lower, atol=1e-10)
        pairs, value = best_term_pairs(w, k, weights=p_c)
        enc = np.eye(pairs.size)[w[:, pairs].argmin(axis=1)]
        _, margins_t, p_t = _strata(margins, p_c, pi, enc)
        b = stratified_tian_pearl(margins_t, p_t).pns
        assert b.upper - b.lower == pytest.approx(value, abs=1e-10)
        narrowest = np.inf
        for labels in product(range(k), repeat=nc):
            enc = np.eye(k)[list(labels)]
            _, margins_t, p_t = _strata(margins, p_c, pi, enc)
            b = stratified_tian_pearl(margins_t, p_t).pns
            narrowest = min(narrowest, b.upper - b.lower)
        assert narrowest == pytest.approx(value, abs=1e-10)


def test_greedy_search_of_term_pairs_matches_enumeration(rng):
    from ibstrata.stratify import best_term_pairs
    for _ in range(20):
        w = rng.random((200, 16)) + rng.random(16)[None, :]
        exact = best_term_pairs(w, 3)[1]
        greedy = best_term_pairs(w, 3, limit=1)[1]
        assert exact <= greedy + 1e-12
        assert greedy <= exact + 0.02
