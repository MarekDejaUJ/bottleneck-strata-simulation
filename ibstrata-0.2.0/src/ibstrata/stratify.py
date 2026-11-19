"""Honest bottleneck stratification of bounds on probabilities of causation.

Data come from a design with randomized arms: ``m`` forced arms that set the treatment
and one free-choice arm. A stratifier is an encoder from pre-treatment features to soft
memberships ``P(t | person)``. The encoder is fitted on one fold of persons and applied to
the other; the stratified margins of the other fold give valid bounds for the population
(Proposition 4 of the method). Two estimates follow: plug-in bounds at the margins of
the strata, and bounds with the binding term of every stratum selected out of fold
(Proposition 5), whose fold bounds are averaged.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from scipy.stats import norm

from . import ib
from .bounds import BinaryBounds, Bounds, Margins, project_compatible, stratified_tian_pearl
from .bounds.arms import arm_index
from .mib import ReachPracticeBottleneck, knn_smoother
from .relevance import SoftmaxModel, fit_softmax


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
class LogisticRelevance:
    """Relevance model by regression: ``P(x, y | s, c)`` from a multinomial logistic
    regression of the observed cell on the features within every design arm, assembled
    with the design probabilities into ``q_c(s, x, y) = pi_s P(x, y | s, c)``."""

    penalty: float = 1.0
    feature_columns: np.ndarray | None = None
    _models: list[SoftmaxModel] | None = None
    _shape: tuple[int, int] = (2, 2)
    _pi: np.ndarray | None = None

    def _features(self, data: Design) -> np.ndarray:
        f = data.features
        return f if self.feature_columns is None else f[:, self.feature_columns]

    def fit(self, data: Design) -> LogisticRelevance:
        feats = self._features(data)
        m, n_y = data.m, data.n_y
        models = []
        for s in range(m + 1):
            sel = data.arm == s
            if not sel.any():
                raise ValueError("a design arm has no units")
            target = data.y[sel] if s < m else data.x[sel] * n_y + data.y[sel]
            models.append(fit_softmax(feats[sel], target, n_y if s < m else m * n_y,
                                      penalty=self.penalty))
        self._models, self._shape, self._pi = models, (m, n_y), data.pi
        return self

    def predict(self, data: Design) -> np.ndarray:
        """Fitted distributions of the arm-augmented outcome, one row per unit, in the
        layout of :mod:`ibstrata.bounds.arms`."""
        feats = self._features(data)
        m, n_y = self._shape
        blocks = [self._pi[s] * self._models[s].predict(feats) for s in range(m + 1)]
        q = np.concatenate(blocks, axis=1)
        q = np.clip(q, 1e-12, None)
        return q / q.sum(axis=1, keepdims=True)

    def effect(self, data: Design, x: int = 1, y: int = 1) -> np.ndarray:
        """Fitted effect ``P(Y_x = y | c) - P(Y_{x'} = y | c)`` of a binary treatment."""
        feats = self._features(data)
        return (self._models[x].predict(feats)[:, y]
                - self._models[1 - x].predict(feats)[:, y])


@dataclass
class BottleneckEncoder:
    """Single-variable bottleneck on estimated relevance distributions.

    The relevance variable is the arm-augmented outcome. ``k`` is the number of strata
    taken from the annealing path (the largest available count not above ``k``, at the
    largest beta of the schedule). ``relevance`` names the relevance model: ``"neighbors"``
    smooths the observed cells over nearest neighbours in feature space, and a new unit
    receives the average assignment of its neighbours; ``"logistic"`` fits a multinomial
    logistic regression within every design arm, and a new unit is assigned by the
    stationary equation of the bottleneck at its own fitted distribution.
    """

    k: int = 3
    neighbors: int | None = None
    betas: Sequence[float] = tuple(np.geomspace(0.5, 3000, 50))
    seed: int = 0
    feature_columns: np.ndarray | None = None
    balanced: bool = True
    alpha: float = 0.05
    weighted: bool = True
    relevance: str = "neighbors"
    penalty: float = 1.0
    _train: np.ndarray | None = None
    _enc: np.ndarray | None = None
    _k_nn: int = 15
    _wf: np.ndarray | None = None
    _model: LogisticRelevance | None = None
    _sol: ib.IBSolution | None = None

    def _features(self, data: Design) -> np.ndarray:
        f = data.features
        return f if self.feature_columns is None else f[:, self.feature_columns]

    def _last_solution(self, path: ib.AnnealingPath) -> ib.IBSolution:
        best = None
        for sol in path.solutions:
            if sol.k <= self.k:
                best = sol
        return best

    def fit(self, data: Design) -> BottleneckEncoder:
        if self.relevance == "logistic":
            self._model = LogisticRelevance(self.penalty, self.feature_columns).fit(data)
            q = self._model.predict(data)
            p = np.full(data.n, 1.0 / data.n)
            path = ib.anneal(p, q, np.asarray(self.betas), k_max=self.k, seed=self.seed)
            self._sol = self._last_solution(path)
            return self
        if self.relevance != "neighbors":
            raise ValueError("relevance must be 'neighbors' or 'logistic'")
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
        best = self._last_solution(path)
        self._train = feats
        self._enc = best.encoder[:, best.p_t > 1e-12]
        return self

    def memberships(self, data: Design) -> np.ndarray:
        if self.relevance == "logistic":
            g = ib.encode(self._sol, self._model.predict(data))
            return g[:, self._sol.p_t > 1e-12]
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


def _lloyd(x: np.ndarray, k: int, starts: int, rng: np.random.Generator) -> np.ndarray:
    """Centers of the best of ``starts`` runs of Lloyd's algorithm from centers drawn at
    random among the rows of ``x``."""
    best, best_inertia = None, np.inf
    for _ in range(starts):
        c = x[rng.choice(x.shape[0], size=k, replace=False)].copy()
        for _ in range(100):
            lab = ((x[:, None, :] - c[None]) ** 2).sum(axis=2).argmin(axis=1)
            new = np.array([x[lab == t].mean(axis=0) if (lab == t).any() else c[t]
                            for t in range(k)])
            if np.allclose(new, c):
                break
            c = new
        inertia = ((x - c[lab]) ** 2).sum()
        if inertia < best_inertia:
            best, best_inertia = c, inertia
    return best


def _nearest(x: np.ndarray, centers: np.ndarray) -> np.ndarray:
    lab = ((x[:, None, :] - centers[None]) ** 2).sum(axis=2).argmin(axis=1)
    return np.eye(centers.shape[0])[lab]


@dataclass
class KMeansEncoder:
    """k-means clustering of the features (Lloyd iterations, best of several starts).

    With ``weighted = False`` the clusters ignore the outcome: the comparison stratifier
    without relevance. With ``weighted = True`` every feature is scaled by the square
    root of its relevance weight, so squared distances weight the features as the
    nearest-neighbour smoother of the bottleneck does.
    """

    k: int = 3
    starts: int = 10
    seed: int = 0
    weighted: bool = False
    _centers: np.ndarray | None = None
    _scale: np.ndarray | None = None

    def _x(self, data: Design) -> np.ndarray:
        return data.features if self._scale is None else data.features * self._scale[None, :]

    def fit(self, data: Design) -> KMeansEncoder:
        rng = np.random.default_rng(self.seed)
        self._scale = None
        if self.weighted:
            w = relevance_weights(data)
            self._scale = np.sqrt(w) if w.sum() > 0 else None
        self._centers = _lloyd(self._x(data), self.k, self.starts, rng)
        return self

    def memberships(self, data: Design) -> np.ndarray:
        return _nearest(self._x(data), self._centers)


@dataclass
class FittedKMeansEncoder:
    """k-means clustering (squared Euclidean distance) of the fitted distributions of
    the arm-augmented outcome from the logistic relevance model."""

    k: int = 3
    starts: int = 10
    seed: int = 0
    penalty: float = 1.0
    _centers: np.ndarray | None = None
    _model: LogisticRelevance | None = None

    def fit(self, data: Design) -> FittedKMeansEncoder:
        rng = np.random.default_rng(self.seed)
        self._model = LogisticRelevance(self.penalty).fit(data)
        self._centers = _lloyd(self._model.predict(data), self.k, self.starts, rng)
        return self

    def memberships(self, data: Design) -> np.ndarray:
        return _nearest(self._model.predict(data), self._centers)


@dataclass
class EffectQuantileEncoder:
    """Quantile groups of the fitted effect ``P(Y_x = y | c) - P(Y_{x'} = y | c)`` from
    the logistic relevance model: a summary of the features through one predicted
    quantity. The cut points are the quantiles of the fitted effects of the fitting
    units."""

    k: int = 3
    x: int = 1
    y: int = 1
    penalty: float = 1.0
    _cuts: np.ndarray | None = None
    _model: LogisticRelevance | None = None

    def fit(self, data: Design) -> EffectQuantileEncoder:
        self._model = LogisticRelevance(self.penalty).fit(data)
        score = self._model.effect(data, self.x, self.y)
        self._cuts = np.quantile(score, np.arange(1, self.k) / self.k)
        return self

    def memberships(self, data: Design) -> np.ndarray:
        score = self._model.effect(data, self.x, self.y)
        return np.eye(self.k)[np.searchsorted(self._cuts, score, side="right")]


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

    def training_memberships(self, data: Design, fold_of: np.ndarray) -> list[np.ndarray]:
        """For every fold encoder, the memberships of the persons the encoder was fitted
        on, in the common labels; the rows of the encoder's own fold are zero."""
        out = []
        for f, enc in enumerate(self.encoders):
            idx = np.flatnonzero(fold_of != f)
            g = np.zeros((data.n, self.k))
            part = enc.memberships(data.subset(idx))
            for t in range(part.shape[1]):
                g[idx, self.maps[f][t]] += part[:, t]
            out.append(g)
        return out


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
    bounds are computed on the fold alone; the fold bounds are averaged (Proposition 4).
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


def pns_terms(x: int = 1, y: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Coefficients of the four lower and the four upper terms of the Tian-Pearl PNS
    bounds on the arm-conditional cell probabilities of a binary design (layout of
    :mod:`ibstrata.bounds.arms`, every arm normalized to one). The lower bound is the
    largest of the lower terms and the upper bound the smallest of the upper terms."""
    xp, yp = 1 - x, 1 - y
    p_yx, p_yxp, p_ypxp = x * 2 + y, xp * 2 + y, xp * 2 + yp
    pxy, pxyp, pxpy, pxpyp = 4 + x * 2 + y, 4 + x * 2 + yp, 4 + xp * 2 + y, 4 + xp * 2 + yp
    lo = np.zeros((4, 8))
    lo[1, [p_yx, p_yxp]] = 1.0, -1.0
    lo[2, [pxy, pxpy, p_yxp]] = 1.0, 1.0, -1.0
    lo[3, [p_yx, pxy, pxpy]] = 1.0, -1.0, -1.0
    hi = np.zeros((4, 8))
    hi[0, p_yx] = 1.0
    hi[1, p_ypxp] = 1.0
    hi[2, [pxy, pxpyp]] = 1.0, 1.0
    hi[3, [p_yx, p_yxp, pxyp, pxpy]] = 1.0, -1.0, 1.0, 1.0
    return lo, hi


def _arm_proportions(g: np.ndarray, onehot: np.ndarray, arm_of_cell: np.ndarray,
                     alpha: float = 0.0, fallback: np.ndarray | None = None) -> np.ndarray:
    """Cell proportions within every design arm for every stratum, shape (strata, cells).
    ``alpha`` adds a pseudo-count per cell; a stratum without weight in an arm takes the
    rows of ``fallback`` for the cells of the arm."""
    counts = g.T @ onehot + alpha
    out = np.zeros_like(counts)
    for s in np.unique(arm_of_cell):
        cells = np.flatnonzero(arm_of_cell == s)
        tot = counts[:, cells].sum(axis=1, keepdims=True)
        block = counts[:, cells] / np.where(tot > 0, tot, 1.0)
        if fallback is not None:
            block = np.where(tot > 0, block, fallback[:, cells])
        out[:, cells] = block
    return out


def _shrunk_proportions(g: np.ndarray, onehot: np.ndarray, arm_of_cell: np.ndarray,
                        weight: float) -> np.ndarray:
    """Cell proportions within every design arm for every stratum, shrunk toward the
    pooled proportions of all persons with the weight of ``weight`` persons per arm."""
    pooled = _arm_proportions(np.ones((onehot.shape[0], 1)), onehot, arm_of_cell)
    counts = g.T @ onehot
    out = np.zeros_like(counts)
    for s in np.unique(arm_of_cell):
        cells = np.flatnonzero(arm_of_cell == s)
        tot = counts[:, cells].sum(axis=1, keepdims=True) + weight
        block = (counts[:, cells] + weight * pooled[:, cells]) / np.where(tot > 0, tot, 1.0)
        out[:, cells] = np.where(tot > 0, block, pooled[:, cells])
    return out


def _pool_by_arm(g: np.ndarray, arm: np.ndarray, m: int, min_arm: float) -> list[list[int]]:
    """Groups of strata merged until every group holds at least ``min_arm`` persons
    (membership weight) of every design arm; the group with the smallest arm weight
    joins the group with the largest total weight."""
    w = np.stack([g[arm == s].sum(axis=0) for s in range(m + 1)], axis=1)
    groups = [[t] for t in range(g.shape[1])]
    while len(groups) > 1:
        gw = np.array([w[grp].sum(axis=0) for grp in groups])
        short = gw.min(axis=1)
        worst = int(short.argmin())
        if short[worst] >= min_arm:
            break
        total = gw.sum(axis=1)
        total[worst] = -np.inf
        groups[int(total.argmax())] += groups[worst]
        del groups[worst]
    return groups


def _merge_columns(g: np.ndarray, groups: list[list[int]]) -> np.ndarray:
    return np.stack([g[:, grp].sum(axis=1) for grp in groups], axis=1)


@dataclass
class SelectedBounds:
    """PNS bounds with binding terms selected out of fold, with their interval.

    ``lower`` and ``upper`` are the averages of the fold bounds. ``ci_lower`` and
    ``ci_upper`` are the ends of the intersection of the fold intervals, each fold
    interval at the level ``1 - (1 - level) / folds``; ``ci_lower > ci_upper`` marks an
    empty intersection. ``avg_ci_lower`` and ``avg_ci_upper`` are the ends of the interval
    built on the averages and their standard errors ``se_lower`` and ``se_upper``, which
    treat the folds as independent.
    """

    lower: float
    upper: float
    se_lower: float
    se_upper: float
    ci_lower: float
    ci_upper: float
    avg_ci_lower: float
    avg_ci_upper: float
    fold_lower: list[float]
    fold_upper: list[float]
    strata: int


def selected_term_bounds(data: Design, memberships: np.ndarray, fold_of: np.ndarray,
                         train_memberships: Sequence[np.ndarray] | None = None,
                         x: int = 1, y: int = 1, min_arm: float = 2.0,
                         shrink: float = 20.0, boot: int = 200, seed: int = 0,
                         level: float = 0.95) -> SelectedBounds:
    """Stratified Tian-Pearl PNS bounds with the binding term of every stratum selected
    out of fold.

    For every fold, the term of the lower bound and the term of the upper bound are
    selected within every stratum from the persons outside the fold, and the selected
    terms are evaluated at the stratum margins of the persons of the fold; the fold
    bounds are averaged. Every selection gives valid bounds, because a single term of a
    maximum lies below the maximum and a single term of a minimum above the minimum.
    Given the persons outside the fold, the fold bounds are smooth functions of sample
    proportions, so the estimates carry no drift of estimated maxima and minima.

    ``memberships[i]`` are the memberships of person ``i`` from an encoder fitted without
    the fold of the person, or fixed strata. ``train_memberships[f]`` holds, for the
    persons outside fold ``f``, the memberships under the encoder of fold ``f`` (default:
    ``memberships``, for strata fixed in advance). The selection reads the stratum
    margins of the persons outside the fold, shrunk toward their pooled margins with the
    weight of ``shrink`` persons per design arm. Strata with fewer than ``min_arm``
    persons of a design arm in a fold are merged.

    Standard errors of the fold bounds come from ``boot`` resamples of the persons of the
    fold within design arms, with memberships and selections held fixed. Given the
    persons outside a fold, the Imbens-Manski interval of the fold at the level
    ``1 - (1 - level) / folds`` covers the parameter with at least that probability, so
    the intersection of the fold intervals covers the parameter with probability at least
    ``level`` whatever the dependence between the folds.
    """
    if data.m != 2 or data.n_y != 2:
        raise ValueError("selected-term bounds need a binary treatment and a binary outcome")
    a_lo, a_hi = pns_terms(x, y)
    arm_of_cell = arm_index(2, 2)
    onehot = np.eye(8)[data.cell()]
    rng = np.random.default_rng(seed)
    parts = []
    for f in np.unique(fold_of):
        test = np.flatnonzero(fold_of == f)
        train = np.flatnonzero(fold_of != f)
        g_train = (memberships if train_memberships is None else train_memberships[f])[train]
        groups = _pool_by_arm(memberships[test], data.arm[test], data.m, min_arm)
        g_test = _merge_columns(memberships[test], groups)
        g_train = _merge_columns(g_train, groups)
        chosen = _shrunk_proportions(g_train, onehot[train], arm_of_cell, shrink)
        sel_lo = (chosen @ a_lo.T).argmax(axis=1)
        sel_hi = (chosen @ a_hi.T).argmin(axis=1)
        props = _arm_proportions(g_test, onehot[test], arm_of_cell, fallback=chosen)
        parts.append((test, g_test, a_lo[sel_lo], a_hi[sel_hi], props))

    def fold_bounds(g, oh, c_lo, c_hi, fallback):
        props = _arm_proportions(g, oh, arm_of_cell, fallback=fallback)
        p_t = g.mean(axis=0)
        return float(p_t @ (props * c_lo).sum(axis=1)), float(p_t @ (props * c_hi).sum(axis=1))

    per_fold = np.array([fold_bounds(g, onehot[test], c_lo, c_hi, props)
                         for test, g, c_lo, c_hi, props in parts])
    n_folds = per_fold.shape[0]
    draws = np.empty((boot, n_folds, 2))
    arms_by_fold = [[np.flatnonzero(data.arm[test] == s) for s in range(data.m + 1)]
                    for test, *_ in parts]
    for b in range(boot):
        for j, ((test, g, c_lo, c_hi, props), arms) in enumerate(zip(parts, arms_by_fold)):
            idx = np.concatenate([rng.choice(a, size=a.size, replace=True) for a in arms])
            draws[b, j] = fold_bounds(g[idx], onehot[test][idx], c_lo, c_hi, props)
    fold_se = draws.std(axis=0, ddof=1) if boot > 1 else np.zeros((n_folds, 2))
    fold_level = 1.0 - (1.0 - level) / n_folds
    fold_ci = np.array([imbens_manski(per_fold[j, 0], per_fold[j, 1], fold_se[j, 0],
                                      fold_se[j, 1], fold_level) for j in range(n_folds)])
    lower, upper = (float(v) for v in per_fold.mean(axis=0))
    se_lo, se_hi = (float(v) for v in np.sqrt((fold_se ** 2).sum(axis=0)) / n_folds)
    avg_ci = imbens_manski(lower, upper, se_lo, se_hi, level)
    return SelectedBounds(lower, upper, se_lo, se_hi,
                          float(fold_ci[:, 0].max()), float(fold_ci[:, 1].min()),
                          avg_ci[0], avg_ci[1],
                          [float(v) for v in per_fold[:, 0]], [float(v) for v in per_fold[:, 1]],
                          max(g.shape[1] for _, g, *_ in parts))


def pair_widths(q: np.ndarray, pi: np.ndarray, x: int = 1, y: int = 1) -> np.ndarray:
    """Widths ``u_j(q) - l_i(q)`` of the sixteen pairs of a lower term ``l_i`` and an upper
    term ``u_j`` of the Tian-Pearl PNS bounds, for rows ``q`` of distributions of the
    arm-augmented outcome of a binary design; column ``4 * i + j`` holds pair ``(i, j)``.
    The width of the bounds at ``q`` is the smallest of the sixteen pair widths."""
    a_lo, a_hi = pns_terms(x, y)
    p = np.atleast_2d(np.asarray(q, dtype=float)) / np.asarray(pi)[arm_index(2, 2)][None, :]
    lo = p @ a_lo.T
    hi = p @ a_hi.T
    return (hi[:, None, :] - lo[:, :, None]).reshape(p.shape[0], 16)


def best_term_pairs(widths: np.ndarray, k: int, weights: np.ndarray | None = None,
                    limit: int = 50_000) -> tuple[np.ndarray, float]:
    """The ``k`` pairs of terms with the smallest mean, over the rows of ``widths``, of
    the smallest width among the chosen pairs, and that mean.

    The mean is the narrowest width that ``k`` strata can give for the population of the
    rows: every stratifier with ``k`` strata is at least as wide, and the stratifier that
    assigns every row to its narrowest chosen pair attains the mean. All sets of ``k``
    pairs are enumerated when their number is at most ``limit``; otherwise pairs are added
    greedily and then exchanged one at a time until no exchange lowers the mean.
    """
    from itertools import combinations
    from math import comb

    w = np.asarray(widths, dtype=float)
    n, n_pairs = w.shape
    k = min(k, n_pairs)
    wt = np.full(n, 1.0 / n) if weights is None else np.asarray(weights, dtype=float)
    wt = wt / wt.sum()

    def value(cols) -> float:
        return float(wt @ w[:, list(cols)].min(axis=1))

    if comb(n_pairs, k) <= limit:
        combos = np.array(list(combinations(range(n_pairs), k)))
        best, best_val = combos[0], np.inf
        step = max(1, int(4e6 // max(1, n * k)))
        for start in range(0, combos.shape[0], step):
            block = combos[start:start + step]
            vals = wt @ w[:, block].min(axis=2)
            j = int(vals.argmin())
            if vals[j] < best_val - 1e-15:
                best, best_val = block[j], float(vals[j])
        return np.array(best), best_val
    chosen: list[int] = []
    for _ in range(k):
        rest = [c for c in range(n_pairs) if c not in chosen]
        chosen.append(min(rest, key=lambda c: value(chosen + [c])))
    improved = True
    while improved:
        improved = False
        for pos in range(k):
            for c in range(n_pairs):
                if c in chosen:
                    continue
                trial = chosen[:pos] + [c] + chosen[pos + 1:]
                if value(trial) < value(chosen) - 1e-12:
                    chosen, improved = trial, True
    return np.array(sorted(chosen)), value(chosen)


@dataclass
class TermPairEncoder:
    """Strata defined by ``k`` pairs of a lower and an upper term of the Tian-Pearl PNS
    bounds: a person belongs to the stratum of the pair with the smallest width at the
    fitted distribution of the person.

    The pairs are the ``k`` pairs with the smallest mean, over the fitting persons, of the
    width of the narrowest chosen pair at the fitted distributions of the logistic
    relevance model. With the exact distributions in the place of the fitted ones, the
    strata are the ``k`` strata with the narrowest PNS bounds.
    """

    k: int = 3
    x: int = 1
    y: int = 1
    penalty: float = 1.0
    _pairs: np.ndarray | None = None
    _model: LogisticRelevance | None = None

    def fit(self, data: Design) -> TermPairEncoder:
        if data.m != 2 or data.n_y != 2:
            raise ValueError("term-pair strata need a binary treatment and a binary outcome")
        self._model = LogisticRelevance(self.penalty).fit(data)
        w = pair_widths(self._model.predict(data), data.pi, self.x, self.y)
        self._pairs, _ = best_term_pairs(w, self.k)
        return self

    @property
    def pairs(self) -> list[tuple[int, int]]:
        """Chosen pairs as (index of the lower term, index of the upper term)."""
        return [(int(c) // 4, int(c) % 4) for c in self._pairs]

    def memberships(self, data: Design) -> np.ndarray:
        w = pair_widths(self._model.predict(data), data.pi, self.x, self.y)[:, self._pairs]
        return np.eye(self._pairs.size)[w.argmin(axis=1)]
