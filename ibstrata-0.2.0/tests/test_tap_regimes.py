"""The tap-order regime generator: its features equal the statistics that
``ibstrata.orders`` computes from the same tapped orders."""

import numpy as np
import pandas as pd
import pytest

from ibstrata.orders import TapTask
from ibstrata.simulate import TapRegimeDGP, TapTaskModel, dominant_type_probs

ITEMS = ["a", "b", "c", "d"]
BLOCKS = [["a", "b", "c"], ["a", "b", "d"], ["a", "c", "d"], ["b", "c", "d"]]


def small_dgp(exclusion=0.0):
    value = TapTaskModel("value", ITEMS, BLOCKS, np.array([[2.0, 1.0, -1.0, -2.0],
                                                            [-2.0, -1.0, 1.0, 2.0]]), "access")
    episode = TapTaskModel("episode", ITEMS[:3], [ITEMS[:3]], np.array([[1.0, 0.0, -1.0],
                                                                        [-1.0, 0.0, 1.0]]),
                           "practice", anchor=None)
    types = dominant_type_probs([["never", "helped"], ["helped", "always"]], 0.85)
    return TapRegimeDGP([value, episode], np.array([0.5, 0.5]), np.array([[0.7, 0.3], [0.3, 0.7]]),
                        types, np.zeros((2, 2)), exclusion=exclusion)


def test_features_equal_the_order_statistics_of_the_same_orders():
    dgp = small_dgp()
    data, a, m, taps = dgp.sample(60, np.random.default_rng(1), orders=True)
    assert data.features.shape == (60, len(dgp.feature_names()))
    col = 0
    for t in dgp.tasks:
        rows = [{"response_uuid": f"p{i:03d}", "position": b, "cards": "|".join(o),
                 "order": o, "gate": "", "attention": False}
                for i, person in enumerate(taps[t.key]) for b, o in enumerate(person)]
        task = TapTask(t.key, t.items, t.anchor, pd.DataFrame(rows))
        expected = task.features().sort_index().to_numpy()
        width = expected.shape[1]
        np.testing.assert_allclose(data.features[:, col:col + width], expected)
        col += width
    groups = dgp.feature_groups()
    assert len(groups["access"]) == 4 + 6 and len(groups["practice"]) == 3


def test_regimes_shape_the_orders_and_the_truth_is_known():
    dgp = small_dgp()
    data, a, m = dgp.sample(2000, np.random.default_rng(2))
    share_a = data.features[:, 0]  # share of blocks with item a before the anchor
    assert share_a[a == 0].mean() > 0.8 > 0.2 > share_a[a == 1].mean()
    truth = dgp.regimes.truth()
    assert 0 < truth["PNS"] < 1
    b = dgp.regimes.oracle_bounds()
    assert b.pns.lower - 1e-12 <= truth["PNS"] <= b.pns.upper + 1e-12
    assert np.mean(a == m) == pytest.approx(0.7, abs=0.05)


def test_exclusion_removes_the_stated_share():
    data, a, m = small_dgp(exclusion=0.1).sample(5000, np.random.default_rng(3))
    assert data.n == a.shape[0] == m.shape[0]
    assert data.n / 5000 == pytest.approx(0.9, abs=0.02)
