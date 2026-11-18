"""Honest bottleneck stratification of bounds on probabilities of causation.

Data come from a design with randomized arms: ``m`` forced arms that set the treatment
and one free-choice arm. A stratifier is an encoder from pre-treatment features to soft
memberships ``P(t | person)``. The encoder is fitted on one fold of persons and applied to
the other; the stratified margins of the other fold give valid bounds for the population
(Proposition 2 of the method), and the bounds of the two folds are averaged.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from scipy.stats import norm

from . import ib
from .bounds import BinaryBounds, Bounds, Margins, project_compatible, stratified_tian_pearl
from .mib import ReachPracticeBottleneck, knn_smoother


@dataclass
class Design:
    """Person-level data of a randomized design with forced arms and a choice arm.

    ``arm[i]`` is ``j`` for the forced arm of treatment ``j`` and ``m`` for the choice arm;
    ``x[i]`` is the received treatment (equal to the arm in forced arms); ``y[i]`` the
    outcome; ``features`` the pre-treatment covariates; ``pi`` the allocation
    probabilities of the ``m + 1`` arms.
    """

    arm: np.ndarray
    x: np.ndarray
    y: np.ndarray
    features: np.ndarray
    m: int = 2
    n_y: int = 2
    pi: np.ndarray | None = None
    groups: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.arm = np.asarray(self.arm, dtype=int)
        self.x = np.asarray(self.x, dtype=int)
        self.y = np.asarray(self.y, dtype=int)
        self.features = np.asarray(self.features, dtype=float)
        n = self.arm.shape[0]
        if not (self.x.shape == self.y.shape == (n,)) or self.features.shape[0] != n:
            raise ValueError("arm, x, y and features must describe the same persons")
        forced = self.arm < self.m
        if (self.x[forced] != self.arm[forced]).any():
            raise ValueError("in a forced arm the received treatment equals the arm")
        if self.pi is None:
            self.pi = np.bincount(self.arm, minlength=self.m + 1) / n
        self.pi = np.asarray(self.pi, dtype=float)

    @property
    def n(self) -> int:
        return self.arm.shape[0]

    @property
    def n_cells(self) -> int:
        return 2 * self.m * self.n_y

    def cell(self) -> np.ndarray:
        """Cell of the arm-augmented outcome V = (S, X, Y) for every person, in the layout
        of :mod:`ibstrata.bounds.arms`."""
        forced = self.arm < self.m
        c = np.where(forced, self.arm * self.n_y + self.y,
                     self.m * self.n_y + self.x * self.n_y + self.y)
        return c.astype(int)

    def relevance(self) -> np.ndarray:
        """Point-mass relevance distributions ``P(v | person)``."""
        return np.eye(self.n_cells)[self.cell()]

    def subset(self, idx: np.ndarray) -> Design:
        return Design(self.arm[idx], self.x[idx], self.y[idx], self.features[idx], self.m,
                      self.n_y, self.pi, {k: v[idx] for k, v in self.groups.items()})

    def margins(self, weights: np.ndarray | None = None, alpha: float = 0.0) -> Margins:
        """Weighted margins; ``alpha`` adds a Dirichlet pseudo-count per cell."""
        w = np.ones(self.n) if weights is None else np.asarray(weights, dtype=float)
        exp = np.full((self.m, self.n_y), alpha)
        obs = np.full((self.m, self.n_y), alpha)
        for j in range(self.m):
            s = self.arm == j
            np.add.at(exp[j], self.y[s], w[s])
        s = self.arm == self.m
        np.add.at(obs, (self.x[s], self.y[s]), w[s])
        if (exp.sum(axis=1) <= 0).any() or obs.sum() <= 0:
            raise ValueError("a stratum has no weight in some arm")
        return Margins(exp / exp.sum(axis=1, keepdims=True), obs / obs.sum())


def smoothed_relevance(data: Design, smoother: np.ndarray, alpha: float = 0.5,
                       balanced: bool = True) -> np.ndarray:
    """Relevance distributions ``P(V | a)`` for the values ``a`` of a smoothed variable.

    ``smoother[i, a]`` links person ``i`` to value ``a``. With ``balanced = True`` the
    distribution is assembled from the design: ``pi_s * P(x, y | s, a)`` with the arm-
    conditional outcome distribution estimated from the persons of arm ``s`` linked to
    ``a`` (Dirichlet ``alpha`` per cell), so random arm imbalance among neighbours does not
    enter the relevance variable; this is the ``q_c`` of the method's Theorem 1.
    """
    r = data.relevance()
    w = np.asarray(smoother, dtype=float)
    col = w / np.maximum(w.sum(axis=0, keepdims=True), 1e-300)  # P(i | a), columns over a
    if not balanced:
        return col.T @ r
    out = np.zeros((w.shape[1], data.n_cells))
    arm_of_cell = np.concatenate([np.repeat(np.arange(data.m), data.n_y),
                                  np.full(data.m * data.n_y, data.m)])
    for sidx in range(data.m + 1):
        cells = np.flatnonzero(arm_of_cell == sidx)
        persons = np.flatnonzero(data.arm == sidx)
        block = col[persons].T @ r[np.ix_(persons, cells)] + alpha
        block /= block.sum(axis=1, keepdims=True)
        out[:, cells] = data.pi[sidx] * block
    return out


def relevance_weights(data: Design, columns: np.ndarray | None = None,
                      max_levels: int = 10) -> np.ndarray:
    """Feature weights ``max(0, I(F; V) - bias)`` with ``V`` the arm-augmented outcome and
    the Miller-Madow bias ``(|F| - 1)(|V| - 1) / (2 n)``; continuous features are cut into
    at most ``max_levels`` quantile bins."""
    feats = data.features if columns is None else data.features[:, columns]
    cells = data.cell()
    out = np.zeros(feats.shape[1])
    for j in range(feats.shape[1]):
        f = feats[:, j]
        levels = np.unique(f)
        if levels.size > max_levels:
            f = np.searchsorted(np.quantile(f, np.linspace(0, 1, max_levels + 1)[1:-1]), f)
            levels = np.unique(f)
        codes = np.searchsorted(levels, f)
        table = np.zeros((levels.size, data.n_cells))
        np.add.at(table, (codes, cells), 1.0)
        table /= table.sum()
        pa, pb = table.sum(1, keepdims=True), table.sum(0, keepdims=True)
        mask = table > 0
        mi = float((table[mask] * np.log(table[mask] / (pa @ pb)[mask])).sum())
        bias = (levels.size - 1) * (np.count_nonzero(pb) - 1) / (2.0 * data.n)
        out[j] = max(0.0, mi - bias)
    return out


def choose_neighbors(data: Design, columns: np.ndarray | None = None,
                     candidates: Sequence[int] = (5, 10, 15, 20, 30, 45, 60)) -> int:
    """Neighbourhood size with the largest leave-one-out log-likelihood of the observed
    cells of ``V``."""
    from .mib import loo_log_likelihood
    feats = data.features if columns is None else data.features[:, columns]
    cands = [k for k in candidates if k < data.n]
    scores = [loo_log_likelihood(feats, data.cell(), data.n_cells, k) for k in cands]
    return int(cands[int(np.argmax(scores))])


class Encoder(Protocol):
    def fit(self, data: Design) -> Encoder: ...
    def memberships(self, data: Design) -> np.ndarray: ...


@dataclass
class BottleneckEncoder:
    """Single-variable bottleneck on nearest-neighbour smoothed relevance.

    The relevance variable is the arm-augmented outcome. ``k`` is the number of strata
    taken from the annealing path (the largest available count not above ``k``).
    """

    k: int = 3
    neighbors: int | None = None
    betas: Sequence[float] = tuple(np.geomspace(0.5, 3000, 50))
    seed: int = 0
    feature_columns: np.ndarray | None = None
    balanced: bool = True
    alpha: float = 0.05
    weighted: bool = True
    _train: np.ndarray | None = None
    _enc: np.ndarray | None = None
    _k_nn: int = 15
    _wf: np.ndarray | None = None

    def _features(self, data: Design) -> np.ndarray:
        f = data.features
        return f if self.feature_columns is None else f[:, self.feature_columns]

    def fit(self, data: Design) -> BottleneckEncoder:
        feats = self._features(data)
        self._k_nn = (max(10, int(round(np.sqrt(data.n)))) if self.neighbors is None
                      else min(self.neighbors, data.n))
        self._wf = relevance_weights(data, self.feature_columns) if self.weighted else None
        if self._wf is not None and self._wf.sum() <= 0:
            self._wf = None
        w = knn_smoother(feats, self._k_nn, weights=self._wf)
        p_a = w.mean(axis=0)
        q_a = smoothed_relevance(data, w, alpha=self.alpha, balanced=self.balanced)
        path = ib.anneal(p_a, q_a, np.asarray(self.betas), k_max=self.k, seed=self.seed)
        best = None
        for sol in path.solutions:
            if sol.k <= self.k:
                best = sol
        self._train = feats
        self._enc = best.encoder[:, best.p_t > 1e-12]
        return self

    def memberships(self, data: Design) -> np.ndarray:
        w = knn_smoother(self._train, self._k_nn, query=self._features(data), weights=self._wf)
        return w @ self._enc


@dataclass
class ReachPracticeEncoder:
    """Reach and practice regimes from two feature groups; strata are their product."""

    access_columns: np.ndarray
    practice_columns: np.ndarray
    k_a: int = 2
    k_m: int = 2
    neighbors: int | None = None
    betas: Sequence[float] = tuple(np.geomspace(0.5, 3000, 50))
    seed: int = 0
    weighted: bool = True
    _fa: np.ndarray | None = None
    _fm: np.ndarray | None = None
    _wa: np.ndarray | None = None
    _wm: np.ndarray | None = None
    _k_nn: int = 15
    _sol: object = None

    def fit(self, data: Design) -> ReachPracticeEncoder:
        fa = data.features[:, self.access_columns]
        fm = data.features[:, self.practice_columns]
        self._k_nn = k = (max(10, int(round(np.sqrt(data.n)))) if self.neighbors is None
                          else min(self.neighbors, data.n))
        if self.weighted:
            self._wa = relevance_weights(data, self.access_columns)
            self._wm = relevance_weights(data, self.practice_columns)
            self._wa = self._wa if self._wa.sum() > 0 else None
            self._wm = self._wm if self._wm.sum() > 0 else None
        model = ReachPracticeBottleneck(data.relevance(), knn_smoother(fa, k, weights=self._wa),
                                        knn_smoother(fm, k, weights=self._wm))
        path = model.anneal(np.asarray(self.betas), self.k_a, self.k_m, seed=self.seed)
        self._sol = path[-1]
        self._fa, self._fm = fa, fm
        return self

    def regime_memberships(self, data: Design) -> tuple[np.ndarray, np.ndarray]:
        """``P(t_A | person)`` and ``P(t_M | person)``."""
        k = self._k_nn
        wa = knn_smoother(self._fa, k, query=data.features[:, self.access_columns],
                          weights=self._wa)
        wm = knn_smoother(self._fm, k, query=data.features[:, self.practice_columns],
                          weights=self._wm)
        return wa @ self._sol.enc_a, wm @ self._sol.enc_m

    def memberships(self, data: Design) -> np.ndarray:
        ga, gm = self.regime_memberships(data)
        return (ga[:, :, None] * gm[:, None, :]).reshape(data.n, -1)


@dataclass
class FixedEncoder:
    """Stratifier given by a known function of the features (oracle regimes, clusters
    fitted elsewhere, or a single stratum)."""

    rule: Callable[[Design], np.ndarray]

    def fit(self, data: Design) -> FixedEncoder:
        return self

    def memberships(self, data: Design) -> np.ndarray:
        lab = np.asarray(self.rule(data))
        if lab.ndim == 2:
            return lab
        k = int(lab.max()) + 1
        return np.eye(k)[lab]


@dataclass
class KMeansEncoder:
    """Outcome-blind k-means clustering of the features (Lloyd iterations, best of
    several starts); the comparison stratifier that ignores relevance."""

    k: int = 3
    starts: int = 10
    seed: int = 0
    _centers: np.ndarray | None = None

    def fit(self, data: Design) -> KMeansEncoder:
        rng = np.random.default_rng(self.seed)
        x = data.features
        best, best_inertia = None, np.inf
        for _ in range(self.starts):
            c = x[rng.choice(data.n, size=self.k, replace=False)].copy()
            for _ in range(100):
                lab = ((x[:, None, :] - c[None]) ** 2).sum(axis=2).argmin(axis=1)
                new = np.array([x[lab == t].mean(axis=0) if (lab == t).any() else c[t]
                                for t in range(self.k)])
                if np.allclose(new, c):
                    break
                c = new
            inertia = ((x - c[lab]) ** 2).sum()
            if inertia < best_inertia:
                best, best_inertia = c, inertia
        self._centers = best
        return self

    def memberships(self, data: Design) -> np.ndarray:
        lab = ((data.features[:, None, :] - self._centers[None]) ** 2).sum(axis=2).argmin(axis=1)
        return np.eye(self.k)[lab]


@dataclass
class StratifiedEstimate:
    """Honest stratified bounds with the per-fold details."""

    bounds: BinaryBounds
    folds: list[BinaryBounds]
    incompatible_strata: int
    strata: int


def _fold_bounds(data: Design, memberships: np.ndarray, x: int, y: int,
                 alpha: float, min_weight: float, project: bool = False
                 ) -> tuple[BinaryBounds, int, int]:
    memberships = pool_small_strata(memberships, min_weight)
    p_t = memberships.mean(axis=0)
    margins, weights, bad = [], [], 0
    for t in range(memberships.shape[1]):
        mg = data.margins(memberships[:, t], alpha=alpha)
        if not mg.is_compatible(tol=0.0):
            bad += 1
            if project:
                w = memberships[:, t]
                n_exp = np.array([w[data.arm == j].sum() for j in range(data.m)])
                mg, _ = project_compatible(mg, n_exp, w[data.arm == data.m].sum())
        margins.append(mg)
        weights.append(p_t[t])
    return stratified_tian_pearl(margins, weights, x, y), bad, len(margins)


def pool_small_strata(memberships: np.ndarray, min_weight: float) -> np.ndarray:
    """Merge strata whose expected size is below ``min_weight`` persons into one stratum,
    and that stratum into the largest one when it is still too small. Merging is a
    coarsening, so the stratified bounds stay valid (Proposition 1)."""
    g = np.asarray(memberships, dtype=float)
    n = g.shape[0]
    size = g.sum(axis=0)
    small = size < min_weight
    if small.sum() == 0:
        return g
    big = g[:, ~small]
    pooled = g[:, small].sum(axis=1, keepdims=True)
    if big.shape[1] == 0:
        return np.ones((n, 1))
    if pooled.sum() < min_weight:
        big[:, int(big.sum(axis=0).argmax())] += pooled[:, 0]
        return big
    return np.concatenate([big, pooled], axis=1)


def _fold_labels(data: Design, folds: int, rng: np.random.Generator) -> np.ndarray:
    fold_of = np.empty(data.n, dtype=int)
    for a in np.unique(data.arm):
        idx = np.flatnonzero(data.arm == a)
        rng.shuffle(idx)
        fold_of[idx] = np.arange(idx.size) % folds
    return fold_of


@dataclass
class CrossFit:
    """Fold encoders whose strata are mapped onto common labels.

    ``encoders[f]`` is fitted without the persons of fold ``f`` and assigns their
    memberships; ``maps[f][t]`` is the common label of stratum ``t`` of encoder ``f``.
    """

    encoders: list
    maps: list[np.ndarray]
    k: int

    def memberships(self, data: Design, fold_of: np.ndarray) -> np.ndarray:
        g = np.zeros((data.n, self.k))
        for f, enc in enumerate(self.encoders):
            idx = np.flatnonzero(fold_of == f)
            if idx.size == 0:
                continue
            part = enc.memberships(data.subset(idx))
            for t in range(part.shape[1]):
                g[idx, self.maps[f][t]] += part[:, t]
        return g


def crossfit(data: Design, encoder: Encoder, folds: int = 2, seed: int = 0,
             align: bool = True) -> tuple[CrossFit, np.ndarray]:
    """Fit one encoder per fold (on the other folds) and align their strata.

    Alignment matches the strata of every encoder to those of the first by the
    agreement of their memberships over all persons (Hungarian assignment on
    ``sum_i P_0(t | i) P_f(t' | i)``); strata without a partner get labels of their own.
    Merging matched strata is a coarsening of the fold-indexed stratifier, so the bounds
    stay valid; the map uses the persons' features through the fitted encoders only.
    """
    import copy

    from scipy.optimize import linear_sum_assignment

    fold_of = _fold_labels(data, folds, np.random.default_rng(seed))
    encs = []
    for f in range(folds):
        enc = copy.deepcopy(encoder)
        encs.append(enc.fit(data.subset(np.flatnonzero(fold_of != f))))
    ref = encs[0].memberships(data)
    k = ref.shape[1]
    maps = [np.arange(ref.shape[1])]
    for f in range(1, folds):
        g = encs[f].memberships(data)
        if not align:
            maps.append(np.arange(k, k + g.shape[1]))
            k += g.shape[1]
            continue
        agree = ref.T @ g
        rows, cols = linear_sum_assignment(-agree)
        m = np.full(g.shape[1], -1)
        m[cols] = rows
        for t in np.flatnonzero(m < 0):
            m[t] = k
            k += 1
        maps.append(m)
    return CrossFit(encs, maps, k), fold_of


def out_of_fold_memberships(data: Design, encoder: Encoder, folds: int = 2,
                            seed: int = 0, align: bool = True) -> np.ndarray:
    """Memberships of every person from the encoder fitted without the person's fold,
    on strata aligned across folds."""
    cf, fold_of = crossfit(data, encoder, folds, seed, align)
    return cf.memberships(data, fold_of)


def stratified_from_memberships(data: Design, memberships: np.ndarray, x: int = 1,
                                y: int = 1, alpha: float = 0.5, min_weight: float = 5.0,
                                bias_correction: int = 0, seed: int = 0,
                                project: bool = False
                                ) -> tuple[BinaryBounds, int, int]:
    """Stratified bounds for fixed memberships; with ``bias_correction = B > 0`` each
    endpoint is corrected by the bootstrap, ``2 * estimate - mean over B resamples``,
    with the memberships held fixed and persons resampled within arms."""
    base, bad, k = _fold_bounds(data, memberships, x, y, alpha, min_weight, project)
    if bias_correction <= 0:
        return base, bad, k
    rng = np.random.default_rng(seed)
    arms = [np.flatnonzero(data.arm == a) for a in np.unique(data.arm)]
    draws = []
    for _ in range(bias_correction):
        idx = np.concatenate([rng.choice(a, size=a.size, replace=True) for a in arms])
        draws.append(_fold_bounds(data.subset(idx), memberships[idx], x, y, alpha,
                                  min_weight, project)[0])

    def corrected(attr):
        b = getattr(base, attr)
        lo = 2 * b.lower - float(np.mean([getattr(d, attr).lower for d in draws]))
        hi = 2 * b.upper - float(np.mean([getattr(d, attr).upper for d in draws]))
        return Bounds(max(lo, 0.0), min(hi, 1.0), "bias-corrected stratified Tian-Pearl")

    out = BinaryBounds(corrected("pns"), corrected("pn"), corrected("ps"),
                       corrected("pn_joint"), corrected("ps_joint"))
    return out, bad, k


def honest_bounds(data: Design, encoder: Encoder, x: int = 1, y: int = 1, folds: int = 2,
                  seed: int = 0, alpha: float = 0.5, min_weight: float = 5.0,
                  mode: str = "pooled", bias_correction: int = 0) -> StratifiedEstimate:
    """Cross-fitted stratified Tian-Pearl bounds.

    ``mode = "fold"``: for each fold the encoder is fitted on the other folds and the
    bounds are computed on the fold alone; the fold bounds are averaged (Proposition 2).
    ``mode = "pooled"``: every person receives the memberships of the encoder fitted
    without the person's fold, and one set of stratified bounds is computed on the whole
    sample. Strata with expected size below ``min_weight`` persons are pooled (a
    coarsening); ``alpha`` adds a Dirichlet pseudo-count per cell of every stratum.
    """
    if mode == "pooled":
        g = out_of_fold_memberships(data, encoder, folds, seed)
        b, bad, k = stratified_from_memberships(data, g, x, y, alpha, min_weight,
                                                bias_correction, seed)
        return StratifiedEstimate(b, [b], bad, k)
    if mode != "fold":
        raise ValueError("mode must be 'pooled' or 'fold'")
    rng = np.random.default_rng(seed)
    fold_of = _fold_labels(data, folds, rng)
    per_fold, bad_total, strata = [], 0, 0
    for f in range(folds):
        train = data.subset(np.flatnonzero(fold_of != f))
        test = data.subset(np.flatnonzero(fold_of == f))
        enc = encoder.fit(train)
        b, bad, k = stratified_from_memberships(test, enc.memberships(test), x, y, alpha,
                                                min_weight, bias_correction, seed + f)
        per_fold.append(b)
        bad_total += bad
        strata = max(strata, k)
    return StratifiedEstimate(_average(per_fold), per_fold, bad_total, strata)


def plain_bounds(data: Design, x: int = 1, y: int = 1, alpha: float = 0.5,
                 project: bool = False) -> BinaryBounds:
    """Unstratified Tian-Pearl bounds from the pooled margins, projected onto the
    compatible set when ``project`` is true and the margins are incompatible."""
    mg = data.margins(alpha=alpha)
    if project and not mg.is_compatible(tol=0.0):
        n_exp = np.array([(data.arm == j).sum() for j in range(data.m)])
        mg, _ = project_compatible(mg, n_exp, (data.arm == data.m).sum())
    return stratified_tian_pearl([mg], [1.0], x, y)


def _average(results: list[BinaryBounds]) -> BinaryBounds:
    def avg(attr):
        lo = float(np.mean([getattr(r, attr).lower for r in results]))
        hi = float(np.mean([getattr(r, attr).upper for r in results]))
        return Bounds(lo, hi, "honest stratified Tian-Pearl")
    return BinaryBounds(avg("pns"), avg("pn"), avg("ps"), avg("pn_joint"), avg("ps_joint"))


def bootstrap(data: Design, estimator: Callable[[Design], BinaryBounds], reps: int = 200,
              seed: int = 0, quantity: str = "pns") -> np.ndarray:
    """Endpoints of ``quantity`` over bootstrap resamples drawn within arms;
    shape ``(reps, 2)``."""
    rng = np.random.default_rng(seed)
    out = np.empty((reps, 2))
    arms = [np.flatnonzero(data.arm == a) for a in np.unique(data.arm)]
    for r in range(reps):
        idx = np.concatenate([rng.choice(a, size=a.size, replace=True) for a in arms])
        b = getattr(estimator(data.subset(idx)), quantity)
        out[r] = (b.lower, b.upper)
    return out


def imbens_manski(lower: float, upper: float, se_lower: float, se_upper: float,
                  level: float = 0.95) -> tuple[float, float]:
    """Confidence interval for a partially identified parameter (Imbens and Manski,
    2004): ``[L - c se_L, U + c se_U]`` with ``c`` solving
    ``Phi(c + (U - L) / max(se_L, se_U)) - Phi(-c) = level``."""
    width = max(upper - lower, 0.0)
    s = max(se_lower, se_upper, 1e-12)
    lo_c, hi_c = 0.0, 10.0
    for _ in range(100):
        mid = 0.5 * (lo_c + hi_c)
        if norm.cdf(mid + width / s) - norm.cdf(-mid) < level:
            lo_c = mid
        else:
            hi_c = mid
    c = hi_c
    return max(0.0, lower - c * se_lower), min(1.0, upper + c * se_upper)
