"""Bounds averaged over strata of a pre-treatment variable.

For a stratifier T that depends on pre-treatment covariates only, randomization holds
within every stratum, so the bounds of each stratum are valid for the stratum and their
average, weighted by P(t), is valid for the population. Joint probabilities are averaged
directly; a conditional probability such as PN = P(y'_{x'}, x, y) / P(x, y) is obtained by
averaging the bounds of its numerator and dividing by the population probability of the
conditioning event.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from .closed_form import BinaryBounds, tian_pearl
from .events import Event
from .lp import _observed_probability, lp_bounds
from .margins import Margins
from .result import Bounds


def _weights(weights: Sequence[float], count: int) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    if w.shape != (count,) or (w < 0).any() or w.sum() <= 0:
        raise ValueError("weights must be nonnegative, one per stratum, with positive sum")
    return w / w.sum()


def stratified(
    bound: Callable[[Margins], Bounds],
    margins_by_stratum: Sequence[Margins],
    weights: Sequence[float],
) -> Bounds:
    """Weighted average of the bounds of a joint probability over strata."""
    w = _weights(weights, len(margins_by_stratum))
    results = [bound(mg) for mg in margins_by_stratum]
    lower = float(sum(wi * r.lower for wi, r in zip(w, results)))
    upper = float(sum(wi * r.upper for wi, r in zip(w, results)))
    return Bounds(lower, upper, f"stratified {results[0].method}")


def stratified_lp(
    margins_by_stratum: Sequence[Margins],
    weights: Sequence[float],
    event: Event,
    given: Event | None = None,
) -> Bounds:
    """Stratified exact bounds on P(event) or P(event | given)."""
    w = _weights(weights, len(margins_by_stratum))
    target = event if given is None else event.conjoin(given)
    joint = [lp_bounds(mg, target) for mg in margins_by_stratum]
    lower = float(sum(wi * r.lower for wi, r in zip(w, joint)))
    upper = float(sum(wi * r.upper for wi, r in zip(w, joint)))
    if given is None:
        return Bounds(lower, upper, "stratified exact LP")
    denom = float(sum(wi * _observed_probability(mg, given)
                      for wi, mg in zip(w, margins_by_stratum)))
    if denom <= 0:
        raise ValueError("the conditioning event has probability zero")
    return Bounds(lower / denom, min(1.0, upper / denom), "stratified exact LP")


def stratified_tian_pearl(
    margins_by_stratum: Sequence[Margins],
    weights: Sequence[float],
    x: int = 1,
    y: int = 1,
) -> BinaryBounds:
    """Stratified Tian-Pearl bounds on PNS, PN and PS (Mueller, Li and Pearl, 2022,
    Theorem 4, for PNS; PN and PS through their numerators)."""
    w = _weights(weights, len(margins_by_stratum))
    per = [tian_pearl(mg, x, y) for mg in margins_by_stratum]

    def avg(attr: str) -> tuple[float, float]:
        return (
            float(sum(wi * getattr(r, attr).lower for wi, r in zip(w, per))),
            float(sum(wi * getattr(r, attr).upper for wi, r in zip(w, per))),
        )

    pns = avg("pns")
    pnj = avg("pn_joint")
    psj = avg("ps_joint")
    pxy = float(sum(wi * mg.observational[x, y] for wi, mg in zip(w, margins_by_stratum)))
    pxpyp = float(sum(wi * mg.observational[1 - x, 1 - y]
                      for wi, mg in zip(w, margins_by_stratum)))
    label = "stratified Tian-Pearl"

    def ratio(b: tuple[float, float], d: float) -> Bounds:
        if d <= 0:
            return Bounds(float("nan"), float("nan"), label)
        return Bounds(max(0.0, b[0] / d), min(1.0, b[1] / d), label)

    return BinaryBounds(
        pns=Bounds(*pns, label),
        pn=ratio(pnj, pxy),
        ps=ratio(psj, pxpyp),
        pn_joint=Bounds(*pnj, label),
        ps_joint=Bounds(*psj, label),
    )


def mix_margins(margins_by_stratum: Sequence[Margins], weights: Sequence[float]) -> Margins:
    """Margins of the pooled population: the P(t)-weighted mixture of stratum margins."""
    w = _weights(weights, len(margins_by_stratum))
    exp = sum(wi * mg.experimental for wi, mg in zip(w, margins_by_stratum))
    obs = sum(wi * mg.observational for wi, mg in zip(w, margins_by_stratum))
    first = margins_by_stratum[0]
    return Margins(exp, obs, x_labels=first.x_labels, y_labels=first.y_labels)
