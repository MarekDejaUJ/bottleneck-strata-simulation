"""Checks of the tap-order statistics on constructed block responses."""

import numpy as np
import pandas as pd
import pytest

from ibstrata.orders import TapTask, kendall_tau_b

ITEMS = ["a", "b", "c", "d"]
BLOCKS = [["a", "b", "c"], ["a", "b", "d"], ["a", "c", "d"], ["b", "c", "d"]]


def task(person_order, anchor_after=2, attention=False):
    """A respondent who taps by a fixed order and places the anchor after the first
    ``anchor_after`` items of the global order."""
    rank = {c: i for i, c in enumerate(person_order)}
    rows = []
    for pos, block in enumerate(BLOCKS):
        order = sorted(block, key=rank.get)
        above = [c for c in order if rank[c] < anchor_after]
        below = [c for c in order if rank[c] >= anchor_after]
        rows.append({"response_uuid": "r1", "position": pos, "cards": "|".join(block),
                     "order": above + ["ANC"] + below, "gate": "yes", "attention": False})
    return TapTask("t", ITEMS, "ANC", pd.DataFrame(rows))


def test_anchor_shares_follow_the_threshold():
    t = task(["c", "a", "d", "b"], anchor_after=2)
    share = t.anchor_share().loc["r1"]
    assert share["c"] == 1.0 and share["a"] == 1.0
    assert share["d"] == 0.0 and share["b"] == 0.0


def test_copeland_order_and_consistency_of_a_transitive_respondent():
    t = task(["c", "a", "d", "b"])
    scores = t.copeland().loc["r1"]
    assert list(scores.sort_values(ascending=False).index) == ["c", "a", "d", "b"]
    assert t.cyclic_triads().loc["r1"] == 0
    assert t.consistency().loc["r1"] == pytest.approx(1.0)


def test_tau_b_against_reference_tiers():
    t = task(["a", "b", "c", "d"])
    assert t.tau_b([["a"], ["b"], ["c"], ["d"]]).loc["r1"] == pytest.approx(1.0)
    assert t.tau_b([["d"], ["c"], ["b"], ["a"]]).loc["r1"] == pytest.approx(-1.0)
    assert 0 < t.tau_b([["a", "b"], ["c", "d"]]).loc["r1"] < 1


def test_kendall_tau_b_with_ties_matches_the_textbook_formula():
    x = np.array([1, 2, 2, 3.0])
    y = np.array([1, 1, 2, 3.0])
    # concordant 4, discordant 0, ties only in x 1, ties only in y 1: 4 / sqrt(5 * 5)
    assert kendall_tau_b(x, y) == pytest.approx(4 / 5)
