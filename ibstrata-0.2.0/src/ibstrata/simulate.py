"""Data-generating processes with known probabilities of causation.

``RegimeDGP`` draws a latent regime, binary pre-treatment features whose probabilities
depend on the regime, a response type (always, never, helped, hurt) for binary treatment
and outcome, and a natural choice of treatment that may depend on the response type
(confounding in the choice arm). Arms are randomized with known probabilities. The
population values of PNS, PN and PS, and the population margins of every regime, follow
from the parameters without sampling.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from .bounds import Margins, mix_margins, stratified_tian_pearl, tian_pearl
from .stratify import Design

# response types: (Y_{x'}, Y_x) with x = 1 the treatment of interest
TYPES = {"never": (0, 0), "helped": (0, 1), "hurt": (1, 0), "always": (1, 1)}
_ORDER = ("never", "helped", "hurt", "always")


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


@dataclass
class RegimeDGP:
    regime_probs: np.ndarray  # (k,)
    feature_probs: np.ndarray  # (k, d)
    type_probs: np.ndarray  # (k, 4) in the order never, helped, hurt, always
    choice_base: np.ndarray  # (k,) logit of choosing x = 1 by regime
    confounding: float  # added to the logit for helped, subtracted for hurt
    pi: np.ndarray  # (3,) forced x' arm, forced x arm, choice arm
    decoy_probs: np.ndarray | None = None  # (n_decoy_clusters, d_decoy) feature probabilities

    @property
    def k(self) -> int:
        return self.regime_probs.shape[0]

    def choice_prob(self) -> np.ndarray:
        """P(X^o = 1 | regime, type), shape (k, 4)."""
        shift = np.array([0.0, 1.0, -1.0, 0.0]) * self.confounding
        return _sigmoid(self.choice_base[:, None] + shift[None, :])

    def regime_margins(self) -> list[Margins]:
        """Population margins of every regime."""
        out = []
        p1 = self.choice_prob()
        for r in range(self.k):
            tp = self.type_probs[r]
            y0 = np.array([TYPES[t][0] for t in _ORDER])
            y1 = np.array([TYPES[t][1] for t in _ORDER])
            exp = np.array([[1 - tp @ y0, tp @ y0], [1 - tp @ y1, tp @ y1]])
            obs = np.zeros((2, 2))
            for ti in range(4):
                c1 = p1[r, ti]
                obs[1, y1[ti]] += tp[ti] * c1
                obs[0, y0[ti]] += tp[ti] * (1 - c1)
            out.append(Margins(exp, obs))
        return out

    def truth(self) -> dict[str, float]:
        """Population PNS, PN and PS for x = 1, y = 1."""
        p1 = self.choice_prob()
        w = self.regime_probs[:, None] * self.type_probs  # (k, 4)
        helped = w[:, 1]
        pns = float(helped.sum())
        # PN = P(helped, X = 1) / P(X = 1, Y = 1); Y = 1 under X = 1 for helped and always
        num_pn = float((helped * p1[:, 1]).sum())
        den_pn = float((w[:, 1] * p1[:, 1] + w[:, 3] * p1[:, 3]).sum())
        # PS = P(helped, X = 0) / P(X = 0, Y = 0); Y = 0 under X = 0 for never and helped
        num_ps = float((helped * (1 - p1[:, 1])).sum())
        den_ps = float((w[:, 0] * (1 - p1[:, 0]) + w[:, 1] * (1 - p1[:, 1])).sum())
        return {"PNS": pns, "PN": num_pn / den_pn, "PS": num_ps / den_ps}

    def oracle_bounds(self):
        """Population bounds stratified by the true regime."""
        return stratified_tian_pearl(self.regime_margins(), self.regime_probs)

    def marginal_bounds(self):
        """Population bounds without covariates."""
        return tian_pearl(mix_margins(self.regime_margins(), self.regime_probs))

    def regime_distributions(self) -> np.ndarray:
        """Distribution of the arm-augmented outcome V in every regime, shape (k, 8)."""
        from .bounds.arms import to_arm_distribution
        return np.array([to_arm_distribution(mg, self.pi) for mg in self.regime_margins()])

    def posterior(self, features: np.ndarray) -> np.ndarray:
        """Exact ``P(regime | features)``; decoy features are independent of the regime
        and are ignored."""
        f = np.asarray(features, dtype=float)[:, : self.feature_probs.shape[1]]
        lp = np.log(np.clip(self.feature_probs, 1e-12, 1 - 1e-12))
        lq = np.log(np.clip(1 - self.feature_probs, 1e-12, 1 - 1e-12))
        loglik = f @ lp.T + (1 - f) @ lq.T + np.log(self.regime_probs)[None, :]
        loglik -= loglik.max(axis=1, keepdims=True)
        post = np.exp(loglik)
        return post / post.sum(axis=1, keepdims=True)

    def exact_relevance(self, features: np.ndarray) -> np.ndarray:
        """Exact ``q_c = P(V | features)`` for every row of ``features``."""
        return self.posterior(features) @ self.regime_distributions()

    def sample(self, n: int, rng: np.random.Generator):
        """Draw ``n`` persons; returns the design and the latent regime of every person."""
        regime = rng.choice(self.k, size=n, p=self.regime_probs)
        feats = (rng.random((n, self.feature_probs.shape[1]))
                 < self.feature_probs[regime]).astype(float)
        if self.decoy_probs is not None:
            decoy = rng.integers(self.decoy_probs.shape[0], size=n)
            extra = (rng.random((n, self.decoy_probs.shape[1]))
                     < self.decoy_probs[decoy]).astype(float)
            feats = np.concatenate([feats, extra], axis=1)
        u = rng.random(n)
        cum = np.cumsum(self.type_probs[regime], axis=1)
        typ = (u[:, None] > cum).sum(axis=1)
        y0 = np.array([TYPES[t][0] for t in _ORDER])[typ]
        y1 = np.array([TYPES[t][1] for t in _ORDER])[typ]
        arm = rng.choice(3, size=n, p=self.pi)
        choice = (rng.random(n) < self.choice_prob()[regime, typ]).astype(int)
        x = np.where(arm == 2, choice, arm)
        y = np.where(x == 1, y1, y0)
        return Design(arm, x, y, feats, m=2, n_y=2, pi=self.pi), regime


def make_regime_dgp(k: int = 3, d_signal: int = 12, d_noise: int = 8, separation: float = 0.6,
                    heterogeneity: float = 1.0, confounding: float = 1.0,
                    pi: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3),
                    helped: Sequence[float] | None = None, hurt: Sequence[float] | None = None,
                    choice_base: Sequence[float] | None = None,
                    type_probs: Sequence[Sequence[float]] | None = None,
                    decoy_clusters: int = 0, d_decoy: int = 0, decoy_separation: float = 0.9,
                    seed: int = 0) -> RegimeDGP:
    """A regime DGP with controlled separation, heterogeneity and confounding.

    Every regime has a binary prototype over the ``d_signal`` signal features; a feature
    equals the prototype with probability ``(1 + separation) / 2``. Noise features have
    probability 1/2 in every regime. The helped share of regime ``r`` spreads
    ``heterogeneity`` times between 0.05 and 0.55 across regimes around 0.30; the remaining
    mass splits among never, always and hurt. ``helped`` and ``hurt`` set the shares per
    regime directly (mixed-sign effects across regimes when some regimes have more hurt
    than helped units); ``choice_base`` sets the regime logits of choosing x = 1;
    ``type_probs`` (rows never, helped, hurt, always) overrides both share arguments.
    ``decoy_clusters`` latent clusters, independent of the regimes and of the response
    types, shift ``d_decoy`` further features with ``decoy_separation``: a strong feature
    structure that carries no information about the effect.
    """
    rng = np.random.default_rng(seed)
    protos = rng.integers(0, 2, size=(k, d_signal))
    while k > 1 and len({tuple(p) for p in protos}) < k:
        protos = rng.integers(0, 2, size=(k, d_signal))
    p_sig = np.where(protos == 1, (1 + separation) / 2, (1 - separation) / 2)
    feature_probs = np.concatenate([p_sig, np.full((k, d_noise), 0.5)], axis=1)
    if type_probs is not None:
        h = np.asarray(type_probs, dtype=float)[:, 1]
        u = np.asarray(type_probs, dtype=float)[:, 2]
    elif helped is None:
        spread = np.linspace(0.05, 0.55, k) if k > 1 else np.array([0.30])
        h = 0.30 + heterogeneity * (spread - 0.30)
    else:
        h = np.asarray(helped, dtype=float)
    if type_probs is None:
        u = 0.05 * (1.0 - h) if hurt is None else np.asarray(hurt, dtype=float)
    rest = 1.0 - h - u
    if (rest < 0).any():
        raise ValueError("helped and hurt shares exceed one in some regime")
    always = 0.5 * rest
    never = rest - always
    tp = (np.column_stack([never, h, u, always]) if type_probs is None
          else np.asarray(type_probs, dtype=float))
    if choice_base is None:
        cb = np.linspace(-0.8, 0.8, k) if k > 1 else np.zeros(1)
    else:
        cb = np.asarray(choice_base, dtype=float)
    decoy = None
    if decoy_clusters > 0 and d_decoy > 0:
        dp = rng.integers(0, 2, size=(decoy_clusters, d_decoy))
        decoy = np.where(dp == 1, (1 + decoy_separation) / 2, (1 - decoy_separation) / 2)
    return RegimeDGP(np.full(k, 1.0 / k), feature_probs, tp, cb, confounding,
                     np.asarray(pi, dtype=float), decoy)


def make_two_regime_dgp(d_signal: int = 8, d_noise: int = 2, separation: float = 0.6,
                        purity: float = 0.85, confounding: float = 1.0,
                        decoy_clusters: int = 0, d_decoy: int = 0,
                        decoy_separation: float = 0.9, seed: int = 0) -> RegimeDGP:
    """Two binary regimes carried by separate feature groups.

    Regime index ``r = 2 * R_A + R_M``. The first ``d_signal + d_noise`` features belong to
    the access group and depend on ``R_A`` only; the next ``d_signal + d_noise`` belong to
    the practice group and depend on ``R_M`` only. Dominant response types for
    (R_A, R_M) = (0, 0), (0, 1), (1, 0), (1, 1) are never, helped, helped, always, each with
    share ``purity``; hurt is 0.03 and the rest splits between the two other types.
    """
    rng = np.random.default_rng(seed)
    pa = rng.integers(0, 2, size=(2, d_signal))
    while (pa[0] == pa[1]).all():
        pa = rng.integers(0, 2, size=(2, d_signal))
    pm = rng.integers(0, 2, size=(2, d_signal))
    while (pm[0] == pm[1]).all():
        pm = rng.integers(0, 2, size=(2, d_signal))
    hi, lo = (1 + separation) / 2, (1 - separation) / 2
    rows = []
    for ra in (0, 1):
        for rm in (0, 1):
            acc = np.concatenate([np.where(pa[ra] == 1, hi, lo), np.full(d_noise, 0.5)])
            pra = np.concatenate([np.where(pm[rm] == 1, hi, lo), np.full(d_noise, 0.5)])
            rows.append(np.concatenate([acc, pra]))
    feature_probs = np.array(rows)
    dominant = ["never", "helped", "helped", "always"]
    tp = []
    for dom in dominant:
        comp = {"never": 0.0, "helped": 0.0, "hurt": 0.03, "always": 0.0}
        comp[dom] = purity
        others = [t for t in ("never", "helped", "always") if t != dom]
        for t in others:
            comp[t] = (0.97 - purity) / 2
        tp.append([comp[t] for t in _ORDER])
    decoy = None
    if decoy_clusters > 0 and d_decoy > 0:
        dp = rng.integers(0, 2, size=(decoy_clusters, d_decoy))
        decoy = np.where(dp == 1, (1 + decoy_separation) / 2, (1 - decoy_separation) / 2)
    return RegimeDGP(np.full(4, 0.25), feature_probs, np.array(tp),
                     np.linspace(-0.8, 0.8, 4), confounding, np.full(3, 1 / 3), decoy)


def purity_compositions(k: int, purity: float) -> np.ndarray:
    """Response-type compositions (never, helped, hurt, always) of ``k`` regimes: with
    ``purity`` 0 every regime has (0.30, 0.30, 0.03, 0.37); otherwise regime ``r`` has the
    dominant type always, helped, never (cycling) with share ``purity``, hurt 0.03, and the
    rest split equally between the two other types."""
    if purity == 0:
        return np.tile([0.30, 0.30, 0.03, 0.37], (k, 1))
    cycle = ["always", "helped", "never"]
    rows = []
    for r in range(k):
        dom = cycle[r % 3]
        comp = {"never": 0.0, "helped": 0.0, "hurt": 0.03, "always": 0.0}
        comp[dom] = purity
        for t in (t for t in ("never", "helped", "always") if t != dom):
            comp[t] = (0.97 - purity) / 2
        rows.append([comp[t] for t in _ORDER])
    return np.array(rows)


def dominant_type_probs(dominant: Sequence[Sequence[str]], purity: float) -> np.ndarray:
    """Type compositions (never, helped, hurt, always) for a table of dominant types: the
    dominant type of a cell has share ``purity``, hurt 0.03, and the rest splits equally
    between the two other types."""
    table = np.empty((len(dominant), len(dominant[0]), 4))
    for i, row in enumerate(dominant):
        for j, dom in enumerate(row):
            comp = {"never": 0.0, "helped": 0.0, "hurt": 0.03, "always": 0.0}
            comp[dom] = purity
            for t in (t for t in ("never", "helped", "always") if t != dom):
                comp[t] = (0.97 - purity) / 2
            table[i, j] = [comp[t] for t in _ORDER]
    return table


# ---------------------------------------------------------------------------------------
# Regimes expressed in tap-order blocks


@dataclass
class TapTaskModel:
    """One tap-order task of ``TapRegimeDGP``.

    ``means[r]`` holds the mean utility of every item in regime ``r`` of the task's group,
    measured from the anchor card (utility 0) in units of the block noise; a person taps the
    cards of a block in decreasing order of perceived utility.
    """

    key: str
    items: list[str]
    blocks: list[list[str]]
    means: np.ndarray  # (regimes of the group, items)
    group: str  # "access" or "practice"
    anchor: str | None = "ANCHOR"


@dataclass
class TapRegimeDGP:
    """Reach and practice regimes that persons express in tap-order blocks.

    A person has a reach regime ``a`` and a practice regime ``m``, drawn from
    ``reach_probs`` and ``practice_given_reach``. In every task of the access group the
    person's utility of an item is ``separation * means[a, item]`` plus a person deviation
    with standard deviation ``person_sd``; the tasks of the practice group use ``m``. In
    every block the perceived utility adds block noise with standard deviation 1 to the
    person's utilities and to the anchor's utility 0, and the person taps the cards in
    decreasing order of perceived utility. The features are the anchor shares and the
    precedence shares of ``ibstrata.orders.TapTask.features``. Response types, choice and
    outcome follow ``RegimeDGP`` on the combined regime ``r = a * k_m + m``; ``exclusion``
    is the share of persons removed at random before the analysis (failed attention
    blocks).
    """

    tasks: list[TapTaskModel]
    reach_probs: np.ndarray  # (k_a,)
    practice_given_reach: np.ndarray  # (k_a, k_m)
    type_probs: np.ndarray  # (k_a, k_m, 4) never, helped, hurt, always
    choice_base: np.ndarray  # (k_a, k_m) logit of choosing x = 1
    confounding: float = 1.0
    pi: np.ndarray = field(default_factory=lambda: np.full(3, 1 / 3))
    separation: float = 1.0
    person_sd: float = 0.5
    exclusion: float = 0.0

    @property
    def k_a(self) -> int:
        return self.reach_probs.shape[0]

    @property
    def k_m(self) -> int:
        return self.practice_given_reach.shape[1]

    @property
    def regimes(self) -> RegimeDGP:
        """The combined regimes as a ``RegimeDGP`` without binary features."""
        probs = (self.reach_probs[:, None] * self.practice_given_reach).ravel()
        k = probs.shape[0]
        return RegimeDGP(probs, np.zeros((k, 0)), self.type_probs.reshape(k, 4),
                         np.asarray(self.choice_base, dtype=float).ravel(), self.confounding,
                         np.asarray(self.pi, dtype=float))

    def feature_names(self) -> list[str]:
        names = []
        for t in self.tasks:
            if t.anchor is not None:
                names += [f"{t.key}:anchor:{c}" for c in t.items]
            names += [f"{t.key}:prec:{a}>{b}" for a, b in _pairs(t.items)]
        return names

    def feature_groups(self) -> dict[str, np.ndarray]:
        """Column indices of the access and the practice features."""
        groups: dict[str, list[int]] = {"access": [], "practice": []}
        col = 0
        for t in self.tasks:
            width = (len(t.items) if t.anchor is not None else 0) + len(_pairs(t.items))
            groups[t.group] += list(range(col, col + width))
            col += width
        return {g: np.array(v, dtype=int) for g, v in groups.items()}

    def sample(self, n: int, rng: np.random.Generator, orders: bool = False):
        """Draw ``n`` persons and remove the excluded share.

        Returns the design and the reach and practice regimes of the analysed persons and,
        with ``orders=True``, the tapped orders as ``{task: [[order of block b] per person]}``.
        """
        design, regime = self.regimes.sample(n, rng)
        a, m = regime // self.k_m, regime % self.k_m
        feats, taps = [], {}
        for t in self.tasks:
            r = a if t.group == "access" else m
            f, o = _tap_features(t, r, self.separation, self.person_sd, rng, orders)
            feats.append(f)
            if orders:
                taps[t.key] = o
        x = np.concatenate(feats, axis=1)
        keep = np.flatnonzero(rng.random(n) >= self.exclusion)
        design = Design(design.arm[keep], design.x[keep], design.y[keep], x[keep], m=2, n_y=2,
                        pi=design.pi)
        if orders:
            taps = {k: [v[i] for i in keep] for k, v in taps.items()}
            return design, a[keep], m[keep], taps
        return design, a[keep], m[keep]


def _pairs(items: Sequence[str]) -> list[tuple[str, str]]:
    return [(items[i], items[j]) for i in range(len(items)) for j in range(i + 1, len(items))]


def _tap_features(task: TapTaskModel, regime: np.ndarray, separation: float, person_sd: float,
                  rng: np.random.Generator, keep_orders: bool):
    """Anchor shares and precedence shares of simulated tap orders, in the column order of
    ``TapTask.features``."""
    items = list(task.items)
    cards = items + ([task.anchor] if task.anchor is not None else [])
    idx = {c: i for i, c in enumerate(cards)}
    n, v = regime.shape[0], len(items)
    util = np.zeros((n, len(cards)))
    util[:, :v] = separation * task.means[regime] + person_sd * rng.standard_normal((n, v))
    wins = np.zeros((n, len(cards), len(cards)))
    meets = np.zeros((len(cards), len(cards)))
    orders = [[] for _ in range(n)] if keep_orders else None
    rows = np.arange(n)
    for block in task.blocks:
        cols = [idx[c] for c in block] + ([idx[task.anchor]] if task.anchor is not None else [])
        perceived = util[:, cols] + rng.standard_normal((n, len(cols)))
        before = perceived[:, :, None] > perceived[:, None, :]
        wins[np.ix_(rows, cols, cols)] += before
        meets[np.ix_(cols, cols)] += 1
        if keep_orders:
            ranked = np.argsort(-perceived, axis=1)
            for i in range(n):
                orders[i].append([cards[cols[j]] for j in ranked[i]])
    parts = []
    if task.anchor is not None:
        an = idx[task.anchor]
        parts.append(wins[:, :v, an] / meets[:v, an])
    pairs = [(idx[a], idx[b]) for a, b in _pairs(items)]
    parts.append(np.stack([wins[:, i, j] / meets[i, j] for i, j in pairs], axis=1))
    return np.concatenate(parts, axis=1), orders
