"""Closed-form bounds on probabilities of causation from experimental and observational margins.

Sources
-------
Tian, J., & Pearl, J. (2000). Probabilities of causation: Bounds and identification.
    Annals of Mathematics and Artificial Intelligence, 28(1-4), 287-313.
Li, A., & Pearl, J. (2024). Probabilities of causation with nonbinary treatment and
    effect. Proceedings of the AAAI Conference on Artificial Intelligence, 38(18),
    20465-20472. Theorems 4-7 (single hypothetical term; sharp).
Shu, X., Wang, S., & Li, A. (2026). Identification of probabilities of causation: From
    recursive to closed-form bounds. Proceedings of IJCAI-26, 4063-4070. Theorem 2.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .margins import Margins
from .result import Bounds


@dataclass(frozen=True)
class BinaryBounds:
    """Bounds on PNS, PN and PS for binary treatment and outcome.

    ``pn_joint`` bounds ``P(y'_{x'}, x, y)`` and ``ps_joint`` bounds ``P(y_x, x', y')``,
    the numerators of PN and PS; stratified analyses average the numerators and divide
    by the population probability of the conditioning cell.
    """

    pns: Bounds
    pn: Bounds
    ps: Bounds
    pn_joint: Bounds
    ps_joint: Bounds


def _binary_terms(margins: Margins, x: int, y: int) -> dict[str, float]:
    if margins.m != 2 or margins.n != 2:
        raise ValueError("binary bounds need two treatment values and two outcome values")
    xp, yp = 1 - x, 1 - y
    exp, obs = margins.experimental, margins.observational
    return {
        "p_yx": exp[x, y],
        "p_yxp": exp[xp, y],
        "p_ypxp": exp[xp, yp],
        "pxy": obs[x, y],
        "pxyp": obs[x, yp],
        "pxpy": obs[xp, y],
        "pxpyp": obs[xp, yp],
        "py": obs[x, y] + obs[xp, y],
    }


def tian_pearl(margins: Margins, x: int = 1, y: int = 1) -> BinaryBounds:
    """Tian-Pearl bounds on PNS = P(y_x, y'_{x'}), PN = P(y'_{x'} | x, y) and
    PS = P(y_x | x', y') for binary X and Y, with ``x`` and ``y`` the indices of the
    treatment and of the outcome of interest."""
    t = _binary_terms(margins, x, y)
    pns_lo = max(0.0, t["p_yx"] - t["p_yxp"], t["py"] - t["p_yxp"], t["p_yx"] - t["py"])
    pns_hi = min(
        t["p_yx"],
        t["p_ypxp"],
        t["pxy"] + t["pxpyp"],
        t["p_yx"] - t["p_yxp"] + t["pxyp"] + t["pxpy"],
    )
    pnj_lo = max(0.0, t["py"] - t["p_yxp"])
    pnj_hi = min(t["pxy"], t["p_ypxp"] - t["pxpyp"])
    psj_lo = max(0.0, t["p_yx"] - t["py"])
    psj_hi = min(t["pxpyp"], t["p_yx"] - t["pxy"])
    pn = _ratio(pnj_lo, pnj_hi, t["pxy"])
    ps = _ratio(psj_lo, psj_hi, t["pxpyp"])
    return BinaryBounds(
        pns=Bounds(pns_lo, pns_hi, "Tian-Pearl"),
        pn=Bounds(*pn, "Tian-Pearl"),
        ps=Bounds(*ps, "Tian-Pearl"),
        pn_joint=Bounds(pnj_lo, pnj_hi, "Tian-Pearl"),
        ps_joint=Bounds(psj_lo, psj_hi, "Tian-Pearl"),
    )


def _ratio(lo: float, hi: float, denom: float) -> tuple[float, float]:
    if denom <= 0:
        return (float("nan"), float("nan"))
    return (max(0.0, lo / denom), min(1.0, hi / denom))


def monotonic_point(margins: Margins, x: int = 1, y: int = 1) -> dict[str, float]:
    """Point values of PNS, PN and PS when the treatment never prevents the outcome
    (``y_{x'}`` implies ``y_x`` for every unit). Under that assumption the Tian-Pearl
    lower bounds are attained: PNS = P(y_x) - P(y_{x'}), PN = [P(y) - P(y_{x'})] / P(x, y),
    PS = [P(y_x) - P(y)] / P(x', y')."""
    t = _binary_terms(margins, x, y)
    return {
        "PNS": t["p_yx"] - t["p_yxp"],
        "PN": (t["py"] - t["p_yxp"]) / t["pxy"] if t["pxy"] > 0 else float("nan"),
        "PS": (t["p_yx"] - t["py"]) / t["pxpyp"] if t["pxpyp"] > 0 else float("nan"),
    }


def monotonicity_violation(margins: Margins, x: int = 1, y: int = 1) -> float:
    """``P(y_{x'}) - P(y_x)``; a positive value contradicts monotonicity in the margins."""
    t = _binary_terms(margins, x, y)
    return float(t["p_yxp"] - t["p_yx"])


# Li and Pearl (2024), Theorems 4-7: one hypothetical term. The functions reproduce the
# printed statements. The printed lower bounds of Theorems 4 and 5 are valid but are not
# attained in general; preservation_sharp and replacement_sharp give the attained values.


def li_pearl_theorem4(margins: Margins, i: int, j: int) -> Bounds:
    """Bounds on P(y_i x_j, y_i), as printed in Li and Pearl (2024, Theorem 4)."""
    e, o = margins.experimental, margins.observational
    lo = max(o[j, i], e[j, i] + o[:, i].sum() - 1.0)
    hi = min(e[j, i], o[:, i].sum())
    return Bounds(lo, hi, "Li-Pearl Theorem 4 (as printed)")


def li_pearl_theorem5(margins: Margins, i: int, j: int, k: int) -> Bounds:
    """Bounds on P(y_i x_j, y_k), i != k, as printed in Li and Pearl (2024, Theorem 5)."""
    if i == k:
        raise ValueError("Theorem 5 needs i != k; use Theorem 4")
    e, o = margins.experimental, margins.observational
    p_yk = o[:, k].sum()
    inner = sum(
        max(0.0, e[j, i] + o[p, k] - 1.0 + o[j].sum() - o[j, i])
        for p in range(margins.m) if p != j
    )
    lo = max(0.0, e[j, i] + p_yk - 1.0, inner)
    hi = min(e[j, i] - o[j, i], p_yk - o[j, k])
    return Bounds(lo, hi, "Li-Pearl Theorem 5 (as printed)")


def preservation_sharp(margins: Margins, i: int, j: int) -> Bounds:
    """Sharp bounds on P(y_i x_j, y_i).

    Units that chose x_j contribute exactly P(x_j, y_i). Among units that chose another
    treatment, the event {Y_{x_j} = y_i} has probability P(y_i x_j) - P(x_j, y_i) and the
    event {Y = y_i} has probability P(y_i) - P(x_j, y_i), inside a region of probability
    1 - P(x_j); their overlap is at least the excess of the two over the region, and an
    allocation of {Y_{x_j} = y_i} to the slack of each arm attains it. The lower bound is
    max{P(x_j, y_i), P(y_i x_j) + P(y_i) - 1 + P(x_j) - P(x_j, y_i)}; the upper bound is
    the printed one.
    """
    e, o = margins.experimental, margins.observational
    px_j = o[j].sum()
    lo = max(o[j, i], e[j, i] + o[:, i].sum() - 1.0 + px_j - o[j, i])
    hi = min(e[j, i], o[:, i].sum())
    return Bounds(lo, hi, "sharp single-term (preservation)")


def replacement_sharp(margins: Margins, i: int, j: int, k: int) -> Bounds:
    """Sharp bounds on P(y_i x_j, y_k), i != k.

    Units that chose x_j cannot belong to the event. Among the others, {Y_{x_j} = y_i}
    has probability P(y_i x_j) - P(x_j, y_i) and {Y = y_k} has probability
    P(y_k) - P(x_j, y_k), inside a region of probability 1 - P(x_j). The lower bound is
    max{0, P(y_i x_j) - P(x_j, y_i) + P(y_k) - P(x_j, y_k) - 1 + P(x_j)}; the upper
    bound is the printed one.
    """
    if i == k:
        raise ValueError("replacement needs i != k; use preservation_sharp")
    e, o = margins.experimental, margins.observational
    px_j = o[j].sum()
    p_yk = o[:, k].sum()
    lo = max(0.0, e[j, i] - o[j, i] + p_yk - o[j, k] - 1.0 + px_j)
    hi = min(e[j, i] - o[j, i], p_yk - o[j, k])
    return Bounds(lo, hi, "sharp single-term (replacement)")


def li_pearl_theorem6(margins: Margins, i: int, j: int, k: int) -> Bounds:
    """Bounds on P(y_i x_j, x_k), j != k."""
    if j == k:
        raise ValueError("Theorem 6 needs j != k; P(y_i x_j, x_j) equals P(x_j, y_i)")
    e, o = margins.experimental, margins.observational
    px = o.sum(axis=1)
    lo = max(0.0, e[j, i] - o[j, i] - 1.0 + px[j] + px[k])
    hi = min(e[j, i] - o[j, i], px[k])
    return Bounds(lo, hi, "Li-Pearl Theorem 6")


def li_pearl_theorem7(margins: Margins, i: int, j: int, k: int, p: int) -> Bounds:
    """Bounds on P(y_i x_j, y_k, x_p), j != p; the general form of PN and PS."""
    if j == p:
        raise ValueError("Theorem 7 needs j != p")
    e, o = margins.experimental, margins.observational
    px = o.sum(axis=1)
    lo = max(0.0, e[j, i] + o[p, k] - 1.0 + px[j] - o[j, i])
    hi = min(e[j, i] - o[j, i], o[p, k])
    return Bounds(lo, hi, "Li-Pearl Theorem 7")


# Shu, Wang and Li (2026), Theorem 2: closed-form bounds on PNS(k).


def shu_wang_li_pns(margins: Margins, pairs: Sequence[tuple[int, int]]) -> Bounds:
    """Closed-form bounds on P(y_{i_1} x_{j_1}, ..., y_{i_k} x_{j_k}).

    ``pairs`` lists ``(j, i)``: treatment index and outcome index, with distinct
    treatment indices. Theorem 2 of Shu, Wang and Li (2026) is stated for
    ``|X| = |Y|`` and outcome indices aligned with treatment indices; each of its terms
    is a Bonferroni or partition argument that holds for distinct treatments and any
    outcome values, which is the form implemented here. The third upper term uses
    ``m + 1`` distinct members of the query for ``m = 1, ..., k - 1``.
    """
    js = [j for j, _ in pairs]
    if len(set(js)) != len(js):
        raise ValueError("treatment indices must be distinct")
    k = len(pairs)
    e, o = margins.experimental, margins.observational
    px = o.sum(axis=1)
    pe = np.array([e[j, i] for j, i in pairs])
    po = np.array([o[j, i] for j, i in pairs])
    ux = np.array([px[j] - o[j, i] for j, i in pairs])  # P(X = x_j, Y != y_i)
    lower_terms = [0.0, pe.sum() - k + 1.0]
    for t in range(k):
        rest = [s for s in range(k) if s != t]
        lower_terms.append(float((pe[rest] + ux[rest]).sum() + po[t] - k + 1.0))
    outside = sum(px[j] for j in range(margins.m) if j not in js)
    upper_terms = [float(po.sum() + outside), *pe.tolist()]
    d = np.sort(pe - po)
    for mm in range(1, k):
        upper_terms.append(float(d[: mm + 1].sum() / mm))
    return Bounds(max(lower_terms), min(upper_terms), "Shu-Wang-Li Theorem 2")
