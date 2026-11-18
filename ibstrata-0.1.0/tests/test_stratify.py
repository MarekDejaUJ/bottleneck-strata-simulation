"""Checks of the honest stratification pipeline and the simulation DGP."""

import numpy as np
import pytest
from scipy.stats import norm

from ibstrata.simulate import make_regime_dgp
from ibstrata.stratify import (
    BottleneckEncoder,
    FixedEncoder,
    KMeansEncoder,
    honest_bounds,
    imbens_manski,
    plain_bounds,
    pool_small_strata,
)


def test_dgp_truth_matches_a_large_sample():
    dgp = make_regime_dgp(k=3, confounding=1.5, seed=4)
    rng = np.random.default_rng(1)
    data, regime = dgp.sample(400_000, rng)
    truth = dgp.truth()
    # PNS from the forced arms under the DGP's regime margins equals the helped share
    assert truth["PNS"] == pytest.approx(dgp.type_probs[:, 1] @ dgp.regime_probs)
    emp = data.margins()
    pop = dgp.marginal_bounds()
    from ibstrata.bounds import tian_pearl
    est = tian_pearl(emp)
    assert est.pns.lower == pytest.approx(pop.pns.lower, abs=0.01)
    assert est.pns.upper == pytest.approx(pop.pns.upper, abs=0.01)


def test_oracle_and_marginal_population_bounds_contain_the_truth():
    for seed in range(20):
        dgp = make_regime_dgp(k=int(1 + seed % 4), confounding=float(seed % 3), seed=seed)
        t = dgp.truth()
        oracle, marginal = dgp.oracle_bounds(), dgp.marginal_bounds()
        for q in ("pns", "pn", "ps"):
            assert getattr(oracle, q).contains(t[q.upper()], tol=1e-9)
            assert getattr(marginal, q).lower <= getattr(oracle, q).lower + 1e-12
            assert getattr(oracle, q).upper <= getattr(marginal, q).upper + 1e-12


def test_honest_bounds_with_oracle_strata_approach_population_oracle():
    dgp = make_regime_dgp(k=3, confounding=1.0, seed=5)
    data, regime = dgp.sample(60_000, np.random.default_rng(2))
    lookup = dict(zip(map(tuple, data.features), regime))
    oracle = FixedEncoder(lambda d: np.array([lookup[tuple(f)] for f in d.features]))
    est = honest_bounds(data, oracle, alpha=0.0)
    pop = dgp.oracle_bounds()
    assert est.bounds.pns.lower == pytest.approx(pop.pns.lower, abs=0.02)
    assert est.bounds.pns.upper == pytest.approx(pop.pns.upper, abs=0.02)


def test_bottleneck_strata_tighten_bounds_and_stay_valid():
    dgp = make_regime_dgp(k=3, separation=0.8, confounding=1.0, seed=6)
    data, _ = dgp.sample(3000, np.random.default_rng(3))
    plain = plain_bounds(data)
    est = honest_bounds(data, BottleneckEncoder(k=3, neighbors=25))
    assert est.bounds.pns.width < plain.pns.width
    t = dgp.truth()["PNS"]
    assert est.bounds.pns.lower - 0.05 <= t <= est.bounds.pns.upper + 0.05


def test_kmeans_encoder_runs():
    dgp = make_regime_dgp(k=2, seed=7)
    data, _ = dgp.sample(800, np.random.default_rng(4))
    est = honest_bounds(data, KMeansEncoder(k=2))
    assert est.strata >= 1


def test_pool_small_strata_is_a_coarsening():
    g = np.array([[0.9, 0.05, 0.05]] * 10 + [[0.1, 0.8, 0.1]] * 10 + [[0.0, 0.0, 1.0]] * 2)
    pooled = pool_small_strata(g, min_weight=5)
    np.testing.assert_allclose(pooled.sum(axis=1), 1.0)
    assert pooled.shape[1] == 2
    assert (pooled.sum(axis=0) >= 5).all()


def test_imbens_manski_critical_values():
    lo, hi = imbens_manski(0.3, 0.3, 0.05, 0.05)
    assert lo == pytest.approx(0.3 - norm.ppf(0.975) * 0.05, abs=1e-4)
    lo, hi = imbens_manski(0.2, 0.6, 0.02, 0.02)
    assert lo == pytest.approx(0.2 - norm.ppf(0.95) * 0.02, abs=1e-4)
    assert hi == pytest.approx(0.6 + norm.ppf(0.95) * 0.02, abs=1e-4)


def test_relevance_weights_select_informative_features():
    from ibstrata.stratify import relevance_weights
    dgp = make_regime_dgp(k=3, separation=0.8, d_signal=6, d_noise=6,
                          type_probs=[[.1, .15, .03, .72], [.1, .72, .03, .15],
                                      [.72, .15, .03, .10]],
                          seed=8)
    data, _ = dgp.sample(3000, np.random.default_rng(9))
    w = relevance_weights(data)
    assert w[:6].mean() > 5 * max(w[6:].mean(), 1e-6)
