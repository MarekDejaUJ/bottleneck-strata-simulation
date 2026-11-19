"""Checks of the multivariate bottleneck: generic Theorem 7.1 engine and the reach-practice
specification."""

import numpy as np
import pytest

from ibstrata.ib import kl_rows
from ibstrata.mib import GenericMIB, ReachPracticeBottleneck, knn_smoother


def _rpr_problem(rng, n=5, nv=3):
    q = rng.dirichlet(np.ones(nv), size=n)
    w_a = rng.dirichlet(np.ones(n), size=n)
    w_m = rng.dirichlet(np.ones(n), size=n)
    p_i = rng.dirichlet(np.ones(n) * 3)
    joint = np.einsum("i,ia,im,iv->amv", p_i, w_a, w_m, q)
    return q, w_a, w_m, p_i, joint


def test_generic_engine_reduces_to_the_original_bottleneck(rng):
    pc_v = rng.dirichlet(np.ones(12)).reshape(4, 3)
    g = GenericMIB(observed={"C": 4, "V": 3}, joint=pc_v,
                   compress={"T": (["C"], 2)}, g_out=[("T", "V")])
    g.initialize(rng)
    d = g.distortion("T")
    full = g.full_joint()  # axes C, V, T
    p_vt = full.sum(axis=0)  # V, T
    dec = (p_vt / p_vt.sum(axis=0, keepdims=True)).T
    cond = pc_v / pc_v.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(d, kl_rows(cond, dec), atol=1e-10)


def test_reach_practice_distortions_match_generic_theorem(rng):
    for _ in range(5):
        q, w_a, w_m, p_i, joint = _rpr_problem(rng)
        rpr = ReachPracticeBottleneck(q, w_a, w_m, p_person=p_i)
        enc_a = rng.dirichlet(np.ones(2), size=5)
        enc_m = rng.dirichlet(np.ones(3), size=5)
        g = GenericMIB(observed={"A": 5, "M": 5, "V": 3}, joint=joint,
                       compress={"TA": (["A"], 2), "TM": (["M"], 3)},
                       g_out=[("TA", "TM"), ("TA", "V"), ("TM", "V")])
        g.encoders = {"TA": enc_a, "TM": enc_m}
        np.testing.assert_allclose(rpr.distortion_a(enc_a, enc_m), g.distortion("TA"), atol=1e-9)
        np.testing.assert_allclose(rpr.distortion_m(enc_a, enc_m), g.distortion("TM"), atol=1e-9)
        assert rpr.lagrangian(enc_a, enc_m, 2.0)[0] == pytest.approx(g.lagrangian(2.0), abs=1e-10)


@pytest.mark.parametrize("beta", [0.5, 0.9])
def test_asynchronous_updates_decrease_the_lagrangian_below_beta_one(rng, beta):
    # Friedman et al. (2001), Theorem 8.1, proved for beta < 1.
    q, w_a, w_m, p_i, _ = _rpr_problem(rng, n=8, nv=4)
    rpr = ReachPracticeBottleneck(q, w_a, w_m, p_person=p_i)
    sol = rpr.iterate(rng.dirichlet(np.ones(3), size=8), rng.dirichlet(np.ones(3), size=8), beta,
                      max_iter=1000, tol=1e-12, record=True)
    assert (np.diff(sol.history) <= 1e-10).all()


def test_smoothing_keeps_practice_out_of_the_reach_regime(rng):
    """Access features carry no information about V; practice features carry all of it.
    With person-identity smoothers the reach regime splits by the outcome; with nearest-
    neighbour smoothers in access space it does not."""
    n = 120
    access = rng.integers(0, 2, size=(n, 6))
    practice_group = rng.integers(0, 2, size=n)
    practice = np.column_stack([practice_group] * 4 + [rng.integers(0, 2, size=(n, 2))])
    v = np.where(rng.random(n) < np.where(practice_group == 1, 0.9, 0.1), 1, 0)
    q = np.eye(2)[v]
    identity = np.eye(n)
    smooth_a = knn_smoother(access, k=15)
    smooth_m = knn_smoother(practice, k=15)
    betas = np.geomspace(1, 60, 25)
    leaky = ReachPracticeBottleneck(q, identity, identity).anneal(betas, 3, 3, seed=2)[-1]
    clean = ReachPracticeBottleneck(q, smooth_a, smooth_m).anneal(betas, 3, 3, seed=2)[-1]

    def info_about_practice(sol):
        hard = sol.person_a.argmax(axis=1)
        table = np.zeros((sol.person_a.shape[1], 2))
        np.add.at(table, (hard, practice_group), 1)
        table /= table.sum()
        pa, pb = table.sum(1, keepdims=True), table.sum(0, keepdims=True)
        mask = table > 0
        return float((table[mask] * np.log(table[mask] / (pa @ pb)[mask])).sum())

    assert info_about_practice(leaky) > 0.3
    assert info_about_practice(clean) < 0.05
    hard_m = clean.person_m.argmax(axis=1)
    purity = sum(np.bincount(practice_group[hard_m == t], minlength=2).max()
                 for t in np.unique(hard_m)) / n
    assert clean.k_m >= 2 and purity > 0.9
