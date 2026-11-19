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


def test_logistic_relevance_approaches_the_exact_conditionals():
    from ibstrata.stratify import LogisticRelevance
    dgp = make_regime_dgp(k=3, separation=0.6, d_signal=8, d_noise=2,
                          type_probs=[[.1, .15, .03, .72], [.1, .72, .03, .15],
                                      [.72, .15, .03, .10]], seed=11)
    data, _ = dgp.sample(6000, np.random.default_rng(12))
    q = LogisticRelevance().fit(data).predict(data)
    np.testing.assert_allclose(q.sum(axis=1), 1.0)
    # every design arm keeps its design probability
    np.testing.assert_allclose(q[:, :2].sum(axis=1), data.pi[0], atol=1e-9)
    np.testing.assert_allclose(q[:, 4:].sum(axis=1), data.pi[2], atol=1e-9)
    assert np.abs(q - dgp.exact_relevance(data.features)).mean() < 0.02


def test_encoders_on_fitted_distributions_give_memberships():
    from ibstrata.stratify import EffectQuantileEncoder, FittedKMeansEncoder, crossfit
    dgp = make_regime_dgp(k=3, separation=0.8, confounding=1.0, seed=6,
                          type_probs=[[.1, .15, .03, .72], [.1, .72, .03, .15],
                                      [.72, .15, .03, .10]])
    data, _ = dgp.sample(1200, np.random.default_rng(3))
    plain = plain_bounds(data)
    for enc in (BottleneckEncoder(k=3, relevance="logistic"), FittedKMeansEncoder(k=3),
                KMeansEncoder(k=3, weighted=True), EffectQuantileEncoder(k=3)):
        cf, fold_of = crossfit(data, enc, folds=2, seed=0)
        g = cf.memberships(data, fold_of)
        assert g.shape[0] == data.n
        np.testing.assert_allclose(g.sum(axis=1), 1.0, atol=1e-9)
        train = cf.training_memberships(data, fold_of)
        for f, gt in enumerate(train):
            assert np.allclose(gt[fold_of == f], 0.0)
            np.testing.assert_allclose(gt[fold_of != f].sum(axis=1), 1.0, atol=1e-9)
        est = honest_bounds(data, enc)
        assert est.bounds.pns.width < plain.pns.width + 0.02


def test_pooling_by_design_arm():
    from ibstrata.stratify import _merge_columns, _pool_by_arm
    arm = np.repeat([0, 1, 2], 6)
    g = np.zeros((18, 3))
    g[:, 0] = 1.0
    g[[0, 6, 12], :] = [0.0, 1.0, 0.0]  # stratum 1: one person per design arm
    g[[1], :] = [0.0, 0.0, 1.0]  # stratum 2: a single person
    groups = _pool_by_arm(g, arm, 2, min_arm=1.0)
    assert sorted(map(sorted, groups)) == [[0, 2], [1]]
    merged = _merge_columns(g, groups)
    np.testing.assert_allclose(merged.sum(axis=1), 1.0)
    assert len(_pool_by_arm(g, arm, 2, min_arm=2.0)) == 1


def test_selected_term_bounds_are_valid_and_free_of_drift():
    """With all four upper terms tied in one profile, the plug-in upper endpoint falls
    below the population endpoint on average, and the endpoint with terms selected out
    of fold stays at or above the population endpoint."""
    from ibstrata.simulate import purity_compositions
    from ibstrata.stratify import _fold_labels, selected_term_bounds, stratified_from_memberships
    dgp = make_regime_dgp(k=3, d_signal=12, d_noise=4, separation=0.5, confounding=1.0,
                          type_probs=purity_compositions(3, 0.85), seed=117)
    pop = dgp.oracle_bounds().pns
    rng = np.random.default_rng(5)
    plug, sel = [], []
    for rep in range(300):
        data, regime = dgp.sample(600, rng)
        g = np.eye(3)[regime]
        fold_of = _fold_labels(data, 2, np.random.default_rng(rep))
        plug.append(stratified_from_memberships(data, g)[0].pns.upper)
        s = selected_term_bounds(data, g, fold_of, boot=2, seed=rep)
        sel.append((s.lower, s.upper))
    sel = np.array(sel)
    se = sel.std(axis=0, ddof=1) / np.sqrt(len(sel))
    assert np.mean(plug) < pop.upper - 0.005
    assert sel[:, 1].mean() > pop.upper - 3 * se[1]
    assert sel[:, 0].mean() < pop.lower + 3 * se[0]
    assert sel[:, 1].mean() < pop.upper + 0.02 and sel[:, 0].mean() > pop.lower - 0.02


def test_intersection_interval_contains_the_population_bounds_in_a_large_sample():
    from ibstrata.simulate import purity_compositions
    from ibstrata.stratify import _fold_labels, selected_term_bounds
    dgp = make_regime_dgp(k=3, d_signal=12, d_noise=4, separation=0.5, confounding=1.0,
                          type_probs=purity_compositions(3, 0.6), seed=117)
    pop = dgp.oracle_bounds().pns
    data, regime = dgp.sample(6000, np.random.default_rng(21))
    fold_of = _fold_labels(data, 2, np.random.default_rng(0))
    s = selected_term_bounds(data, np.eye(3)[regime], fold_of, boot=200, seed=0)
    assert s.ci_lower <= pop.lower <= pop.upper <= s.ci_upper
    assert s.avg_ci_lower <= pop.lower <= pop.upper <= s.avg_ci_upper
    assert s.ci_upper - s.ci_lower >= s.avg_ci_upper - s.avg_ci_lower - 0.02
    assert s.lower == pytest.approx(np.mean(s.fold_lower))
    assert s.upper == pytest.approx(np.mean(s.fold_upper))


def test_term_pair_encoder_assigns_every_person_to_one_pair():
    from ibstrata.stratify import TermPairEncoder, crossfit, pair_widths
    dgp = make_regime_dgp(k=3, separation=0.8, confounding=1.0, seed=6,
                          type_probs=[[.1, .15, .03, .72], [.1, .72, .03, .15],
                                      [.72, .15, .03, .10]])
    data, _ = dgp.sample(1500, np.random.default_rng(3))
    enc = TermPairEncoder(k=3).fit(data)
    assert len(enc.pairs) == 3 and all(0 <= i < 4 and 0 <= j < 4 for i, j in enc.pairs)
    g = enc.memberships(data)
    assert g.shape == (1500, 3)
    np.testing.assert_allclose(g.sum(axis=1), 1.0)
    # the width of the bounds at a distribution is the smallest pair width
    from ibstrata.bounds import tian_pearl
    from ibstrata.bounds.arms import to_arm_distribution
    for mg in dgp.regime_margins():
        w = pair_widths(to_arm_distribution(mg, dgp.pi), dgp.pi)
        b = tian_pearl(mg).pns
        assert w.min() == pytest.approx(b.upper - b.lower, abs=1e-12)
    cf, fold_of = crossfit(data, TermPairEncoder(k=3), folds=2, seed=0)
    assert cf.memberships(data, fold_of).shape[0] == data.n
