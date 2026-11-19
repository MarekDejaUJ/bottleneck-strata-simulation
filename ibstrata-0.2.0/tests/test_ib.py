"""Checks of the single-variable bottleneck algorithms."""

import numpy as np
import pytest

from ibstrata import ib
from ibstrata.info import mutual_information


def planted(rng, groups=3, per=20, nv=6, noise=0.02):
    centers = rng.dirichlet(np.full(nv, 0.4), size=groups)
    rows, labels = [], []
    for g in range(groups):
        for _ in range(per):
            r = centers[g] + noise * rng.dirichlet(np.ones(nv))
            rows.append(r / r.sum())
            labels.append(g)
    q = np.array(rows)
    p_c = np.full(len(rows), 1.0 / len(rows))
    return p_c, q, np.array(labels)


def same_partition(a, b):
    """True when two labelings define the same partition."""
    pairs = {}
    for x, y in zip(a, b):
        if pairs.setdefault(x, y) != y:
            return False
    return len(set(pairs.values())) == len(pairs)


def test_lagrangian_never_increases_and_fixed_point(rng):
    p_c, q, _ = planted(rng)
    enc = rng.dirichlet(np.ones(4), size=q.shape[0])
    sol = ib.iterate(p_c, q, enc, beta=5.0)
    hist = np.array(sol.history)
    assert (np.diff(hist) <= 1e-10).all()
    again = ib.iterate(p_c, q, sol.encoder, beta=5.0, max_iter=1)
    np.testing.assert_allclose(again.encoder, sol.encoder, atol=1e-8)


def test_relevance_respects_data_processing(rng):
    p_c, q, _ = planted(rng)
    full = mutual_information(p_c[:, None] * q)
    for beta in [0.5, 2.0, 20.0]:
        sol = ib.iterate(p_c, q, rng.dirichlet(np.ones(5), size=q.shape[0]), beta)
        assert sol.relevance <= full + 1e-10
        assert sol.compression >= -1e-12


def test_annealing_recovers_planted_strata(rng):
    p_c, q, labels = planted(rng)
    path = ib.anneal(p_c, q, betas=np.geomspace(0.5, 200, 40), k_max=6, seed=1)
    three = path.at_k(3)
    assert three is not None
    assert same_partition(three.hard(), labels)
    assert path.bifurcations()[0][1] == 2


def test_agglomerative_recovers_planted_strata(rng):
    p_c, q, labels = planted(rng)
    tree = ib.agglomerative(p_c, q)
    assert same_partition(tree.labels_by_k[3], labels)
    rel = [tree.relevance_by_k[k] for k in sorted(tree.relevance_by_k)]
    assert all(b >= a - 1e-12 for a, b in zip(rel, rel[1:]))


def test_merge_loss_equals_relevance_drop(rng):
    p_c, q, _ = planted(rng, per=5)
    tree = ib.agglomerative(p_c, q)
    n = q.shape[0]
    for step, (_, _, loss) in enumerate(tree.merges):
        k_before, k_after = n - step, n - step - 1
        drop = tree.relevance_by_k[k_before] - tree.relevance_by_k[k_after]
        assert loss == pytest.approx(drop, abs=1e-10)


def test_deterministic_bottleneck_recovers_planted_strata(rng):
    p_c, q, labels = planted(rng)
    best = None
    for seed in range(10):
        sol = ib.deterministic(p_c, q, k=3, beta=50.0, seed=seed)
        if best is None or sol.lagrangian < best.lagrangian:
            best = sol
    assert same_partition(best.hard(), labels)


def test_encode_reproduces_training_memberships(rng):
    p_c, q, _ = planted(rng)
    sol = ib.iterate(p_c, q, rng.dirichlet(np.ones(3), size=q.shape[0]), beta=8.0)
    np.testing.assert_allclose(ib.encode(sol, q), sol.encoder, atol=1e-7)
