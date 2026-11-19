"""Verification of the bound formulas against the exact linear program and published examples."""

import itertools

import numpy as np
import pytest
from _helpers import margins_and_joint

from ibstrata.bounds import (
    Event,
    Margins,
    event_probability,
    li_pearl_theorem4,
    li_pearl_theorem5,
    li_pearl_theorem6,
    li_pearl_theorem7,
    lp_bounds,
    monotonic_point,
    preservation_sharp,
    replacement_sharp,
    shu_wang_li_pns,
    tian_pearl,
)

TOL = 1e-7


@pytest.mark.parametrize("m,n", [(2, 2), (2, 3), (3, 2), (3, 3)])
def test_lp_bounds_contain_the_true_probability(rng, m, n):
    for _ in range(60):
        margins, joint = margins_and_joint(rng, m, n)
        chosen = rng.choice(m, size=int(rng.integers(1, m + 1)), replace=False)
        do = {int(j): int(rng.integers(n)) for j in chosen}
        event = Event(do=do, x=int(rng.integers(m)) if rng.random() < 0.3 else None)
        b = lp_bounds(margins, event)
        assert b.contains(event_probability(joint, event), tol=1e-8)


def test_from_joint_margins_are_compatible(rng):
    for m, n in [(2, 2), (3, 3), (4, 2)]:
        margins, _ = margins_and_joint(rng, m, n)
        assert margins.is_compatible()


def test_incompatible_margins_are_refused():
    exp = np.array([[0.9, 0.1], [0.5, 0.5]])
    obs = np.array([[0.1, 0.4], [0.25, 0.25]])  # P(Y_{x1} = y2) = 0.1 < P(x1, y2) = 0.4
    margins = Margins(exp, obs)
    assert not margins.is_compatible()
    with pytest.raises(ValueError):
        lp_bounds(margins, Event(do={0: 0, 1: 1}))


def test_tian_pearl_equals_exact_lp(rng):
    for _ in range(400):
        margins, _ = margins_and_joint(rng, 2, 2)
        x, y = int(rng.integers(2)), int(rng.integers(2))
        tp = tian_pearl(margins, x=x, y=y)
        pns = lp_bounds(margins, Event(do={x: y, 1 - x: 1 - y}))
        pn = lp_bounds(margins, Event(do={1 - x: 1 - y}), given=Event(x=x, y=y))
        ps = lp_bounds(margins, Event(do={x: y}), given=Event(x=1 - x, y=1 - y))
        for closed, exact in [(tp.pns, pns), (tp.pn, pn), (tp.ps, ps)]:
            assert closed.lower == pytest.approx(exact.lower, abs=TOL)
            assert closed.upper == pytest.approx(exact.upper, abs=TOL)


def test_monotonic_point_is_attained_under_monotonicity(rng):
    # Joint with no unit for which the treatment prevents the outcome.
    for _ in range(100):
        joint = rng.dirichlet(np.full(8, 0.7)).reshape(2, 2, 2)
        joint[:, 1, 0] = 0.0  # Y_{x0} = 1 and Y_{x1} = 0 excluded
        joint /= joint.sum()
        margins = Margins.from_joint(joint)
        point = monotonic_point(margins, x=1, y=1)
        true_pns = event_probability(joint, Event(do={1: 1, 0: 0}))
        assert point["PNS"] == pytest.approx(true_pns, abs=1e-12)
        p_xy = margins.observational[1, 1]
        true_pn = event_probability(joint, Event(do={0: 0}, x=1, y=1)) / p_xy
        assert point["PN"] == pytest.approx(true_pn, abs=1e-12)


@pytest.mark.parametrize("m,n", [(2, 2), (2, 3), (3, 2), (3, 3), (4, 3)])
def test_single_term_bounds_against_exact_lp(rng, m, n):
    """Theorems 6 and 7 of Li and Pearl (2024) are sharp as printed; the printed lower
    bounds of Theorems 4 and 5 are valid, and the derived lower bounds are attained."""
    for _ in range(60):
        margins, _ = margins_and_joint(rng, m, n, concentration=float(rng.choice([0.3, 1.0, 3.0])))
        i, k = (int(v) for v in rng.choice(n, size=2, replace=False))
        j, p = (int(v) for v in rng.choice(m, size=2, replace=False))
        e4 = lp_bounds(margins, Event(do={j: i}, y=i))
        e5 = lp_bounds(margins, Event(do={j: i}, y=k))
        e6 = lp_bounds(margins, Event(do={j: i}, x=p))
        e7 = lp_bounds(margins, Event(do={j: i}, x=p, y=k))
        for closed, exact in [(preservation_sharp(margins, i, j), e4),
                              (replacement_sharp(margins, i, j, k), e5),
                              (li_pearl_theorem6(margins, i, j, p), e6),
                              (li_pearl_theorem7(margins, i, j, k, p), e7)]:
            assert closed.lower == pytest.approx(exact.lower, abs=TOL), closed.method
            assert closed.upper == pytest.approx(exact.upper, abs=TOL), closed.method
        for printed, exact in [(li_pearl_theorem4(margins, i, j), e4),
                               (li_pearl_theorem5(margins, i, j, k), e5)]:
            assert printed.lower <= exact.lower + TOL
            assert printed.upper == pytest.approx(exact.upper, abs=TOL)


def test_printed_theorem4_lower_bound_is_not_attained_in_a_binary_example():
    # P(y_x) = 0.9, P(x) = 0.5, P(x, y) = 0.45, P(y) = 0.6: printed 0.50, attained 0.55.
    margins = Margins(experimental=[[0.5, 0.5], [0.1, 0.9]],
                      observational=[[0.35, 0.15], [0.05, 0.45]])
    printed = li_pearl_theorem4(margins, i=1, j=1)
    exact = lp_bounds(margins, Event(do={1: 1}, y=1))
    assert printed.lower == pytest.approx(0.50)
    assert exact.lower == pytest.approx(0.55, abs=1e-9)
    assert preservation_sharp(margins, i=1, j=1).lower == pytest.approx(0.55)


@pytest.mark.parametrize("m,n,k", [(2, 3, 2), (3, 3, 2), (3, 3, 3), (4, 3, 3), (4, 4, 4)])
def test_shu_wang_li_bounds_contain_exact_lp(rng, m, n, k):
    for _ in range(30):
        margins, _ = margins_and_joint(rng, m, n)
        js = rng.choice(m, size=k, replace=False)
        pairs = [(int(j), int(rng.integers(n))) for j in js]
        closed = shu_wang_li_pns(margins, pairs)
        exact = lp_bounds(margins, Event(do=dict(pairs)))
        assert closed.lower <= exact.lower + TOL
        assert closed.upper >= exact.upper - TOL


# Worked examples of Li and Pearl (2024), pp. 20469-20470.

CANCER_EXP = [[80, 7, 213], [184, 29, 87], [87, 189, 24]]
CANCER_OBS = [[238, 20, 7], [10, 77, 259], [147, 72, 70]]
JOB_EXP = [[53, 247], [269, 31], [234, 66], [151, 149]]
JOB_OBS = [[92, 58], [55, 118], [24, 231], [599, 23]]
VACCINE_EXP = [[205, 46, 343, 6], [27, 122, 87, 364]]
VACCINE_OBS = [[6, 74, 632, 5], [52, 243, 147, 41]]


def test_cancer_example_pns3_lies_within_published_recursive_bounds():
    margins = Margins.from_counts(CANCER_EXP, CANCER_OBS)
    assert margins.is_compatible()
    exact = lp_bounds(margins, Event(do={0: 2, 1: 0, 2: 1}))
    # Published recursive (Theorem 8) bounds: [0, 0.099]; the exact interval is inside.
    assert exact.lower >= -TOL
    assert exact.upper <= 0.099 + 5e-4


def test_job_example_matches_theorem7():
    margins = Margins.from_counts(JOB_EXP, JOB_OBS)
    assert margins.is_compatible()
    given = Event(x=1, y=1)
    b3 = lp_bounds(margins, Event(do={2: 0}), given=given)
    b4 = lp_bounds(margins, Event(do={3: 0}), given=given)
    assert b3.lower == pytest.approx(0.720, abs=5e-4) and b3.upper == pytest.approx(1.0, abs=5e-4)
    assert b4.lower == pytest.approx(0.0, abs=5e-4) and b4.upper == pytest.approx(0.042, abs=5e-4)
    t7 = li_pearl_theorem7(margins, 0, 2, 1, 1)
    assert t7.lower / margins.observational[1, 1] == pytest.approx(b3.lower, abs=TOL)


def test_vaccine_example_pns2_lies_within_published_recursive_bounds():
    margins = Margins.from_counts(VACCINE_EXP, VACCINE_OBS)
    assert margins.is_compatible()
    published = {0: (0.0, 0.039), 1: (0.037, 0.077), 2: (0.502, 0.561)}
    for i, (lo, hi) in published.items():
        exact = lp_bounds(margins, Event(do={0: i, 1: 3}))
        assert exact.lower >= lo - 5e-4
        assert exact.upper <= hi + 5e-4


def test_event_enumeration_matches_shape():
    for m, n in [(2, 2), (3, 2)]:
        cells = list(itertools.product(range(m), itertools.product(range(n), repeat=m)))
        assert len(cells) == m * n**m


def test_projection_restores_compatibility_and_keeps_compatible_margins(rng):
    from ibstrata.bounds import project_compatible
    margins, _ = margins_and_joint(rng, 2, 2)
    same, moved = project_compatible(margins, 100, 100)
    assert moved == 0.0 and same is margins
    bad = Margins(np.array([[0.7, 0.3], [0.2, 0.8]]), np.array([[0.20, 0.15], [0.25, 0.40]]))
    assert not bad.is_compatible()
    fixed, moved = project_compatible(bad, np.array([50, 50]), 50)
    assert fixed.is_compatible(tol=1e-9) and moved > 0
    b = tian_pearl(fixed)
    assert b.pns.lower <= b.pns.upper + 1e-12 and 0 <= b.pn.lower <= b.pn.upper <= 1
