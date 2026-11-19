"""Multivariate information bottleneck (Friedman, Mosenzon, Slonim and Tishby, 2001).

Two implementations share one definition. ``GenericMIB`` holds the full joint tensor
of the observed and the compression variables and evaluates the distortion of Theorem
7.1 term by term for any pair of networks ``G_in`` and ``G_out``; the tensor grows with
the product of the cardinalities, so the class serves small problems and verification.
``ReachPracticeBottleneck`` implements one specification at the scale of a survey:

* ``G_in``: the reach regime ``T_A`` compresses the access configuration ``A`` and the
  practice regime ``T_M`` compresses the practice configuration ``M``;
* ``G_out``: ``T_A -> T_M`` and ``(T_A, T_M) -> V``, with ``V`` the relevance variable;
* Lagrangian ``I(T_A; A) + I(T_M; M) - beta [I(T_A; T_M) + I(V; T_A, T_M)]``.

Persons are the items. Every person carries an observed relevance distribution (a point
mass on the observed cell of ``V``, or a fitted distribution). ``A`` and ``M`` are linked
to persons by two row-stochastic smoothers ``W_A`` and ``W_M`` built from access features
and from practice features: ``P(i, a, m, v) = P(i) W_A[i, a] W_M[i, m] P(v | i)``. The
smoothers make the joint a proper distribution in which ``A`` carries only what the access
features of a person's neighbours carry, so ``T_A`` cannot read the person's own practice
through the degenerate empirical joint of person-unique configurations.
"""

from __future__ import annotations

import itertools
import string
from dataclasses import dataclass, field

import networkx as nx
import numpy as np

from .ib import kl_rows
from .info import entropy

_FLOOR = 1e-300


# ---------------------------------------------------------------------------
# Smoothers


def knn_smoother(features: np.ndarray, k: int, query: np.ndarray | None = None,
                 metric: str = "l1", weights: np.ndarray | None = None) -> np.ndarray:
    """Row-stochastic k-nearest-neighbour smoother.

    Row ``r`` puts mass ``1/k`` on the ``k`` training persons nearest to query ``r``
    (training persons when ``query`` is None, each person counting as its own nearest
    neighbour). Ties are broken by person index. Distances: ``l1`` (Hamming for binary
    features) or ``l2``, with optional nonnegative feature ``weights``.
    """
    train = np.asarray(features, dtype=float)
    q = train if query is None else np.asarray(query, dtype=float)
    if weights is not None:
        wf = np.sqrt(np.asarray(weights, dtype=float)) if metric == "l2" else np.asarray(
            weights, dtype=float)
        train = train * wf[None, :]
        q = q * wf[None, :]
    n = train.shape[0]
    if not 1 <= k <= n:
        raise ValueError("k must lie between 1 and the number of training persons")
    if metric not in ("l1", "l2"):
        raise ValueError("metric must be 'l1' or 'l2'")
    w = np.zeros((q.shape[0], n))
    rows = max(1, int(2e7 // max(1, n * train.shape[1])))
    for start in range(0, q.shape[0], rows):
        block = q[start:start + rows]
        diff = block[:, None, :] - train[None, :, :]
        d = (np.abs(diff).sum(axis=2) if metric == "l1"
             else np.sqrt((diff ** 2).sum(axis=2)))
        order = np.lexsort((np.broadcast_to(np.arange(n), d.shape), d), axis=1)[:, :k]
        np.put_along_axis(w[start:start + rows], order, 1.0 / k, axis=1)
    return w


def loo_log_likelihood(features: np.ndarray, observed: np.ndarray, n_cells: int, k: int,
                       alpha: float = 0.5, metric: str = "l1") -> float:
    """Leave-one-out log-likelihood of each person's observed cell of ``V`` under the
    smoothed distribution of its ``k`` nearest other persons (Dirichlet ``alpha`` added),
    used to choose ``k``."""
    train = np.asarray(features, dtype=float)
    obs = np.asarray(observed, dtype=int)
    n = train.shape[0]
    w = knn_smoother(train, min(k + 1, n), metric=metric)
    np.fill_diagonal(w, 0.0)
    w /= np.maximum(w.sum(axis=1, keepdims=True), _FLOOR)
    onehot = np.eye(n_cells)[obs]
    pred = (w @ onehot + alpha / n) / (1.0 + alpha * n_cells / n)
    pred /= pred.sum(axis=1, keepdims=True)
    return float(np.log(np.maximum(pred[np.arange(n), obs], _FLOOR)).sum())


# ---------------------------------------------------------------------------
# Generic engine (verification scale)


@dataclass
class GenericMIB:
    """Full-tensor multivariate bottleneck for small discrete problems.

    Parameters
    ----------
    observed
        Ordered mapping from observed variable names to cardinalities.
    joint
        Joint distribution of the observed variables, axes in the order of ``observed``.
    compress
        Mapping from compression variable names to ``(parents in G_in, cardinality)``.
    g_out
        Edges of ``G_out`` among observed and compression variables.
    """

    observed: dict[str, int]
    joint: np.ndarray
    compress: dict[str, tuple[list[str], int]]
    g_out: list[tuple[str, str]]
    encoders: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.names = list(self.observed) + list(self.compress)
        if len(self.names) > len(string.ascii_letters):
            raise ValueError("too many variables")
        self.letter = {v: string.ascii_letters[i] for i, v in enumerate(self.names)}
        self.card = dict(self.observed)
        for t, (_, k) in self.compress.items():
            self.card[t] = k
        self.out = nx.DiGraph()
        self.out.add_nodes_from(self.names)
        self.out.add_edges_from(self.g_out)
        if not nx.is_directed_acyclic_graph(self.out):
            raise ValueError("G_out must be acyclic")
        joint = np.asarray(self.joint, dtype=float)
        if joint.shape != tuple(self.observed.values()):
            raise ValueError("joint shape does not match the observed cardinalities")
        self.joint = joint / joint.sum()

    def initialize(self, rng: np.random.Generator) -> None:
        for t, (parents, k) in self.compress.items():
            size = int(np.prod([self.observed[p] for p in parents]))
            self.encoders[t] = rng.dirichlet(np.ones(k), size=size)

    # joint over every variable, axes in self.names order
    def full_joint(self) -> np.ndarray:
        obs_letters = "".join(self.letter[v] for v in self.observed)
        operands = [self.joint]
        subs = [obs_letters]
        for t, (parents, k) in self.compress.items():
            shape = [self.observed[p] for p in parents] + [k]
            operands.append(self.encoders[t].reshape(shape))
            subs.append("".join(self.letter[p] for p in parents) + self.letter[t])
        out = "".join(self.letter[v] for v in self.names)
        return np.einsum(",".join(subs) + "->" + out, *operands)

    def marginal(self, full: np.ndarray, keep: list[str]) -> np.ndarray:
        src = "".join(self.letter[v] for v in self.names)
        dst = "".join(self.letter[v] for v in keep)
        return np.einsum(f"{src}->{dst}", full)

    def _mi(self, full: np.ndarray, a: list[str], b: list[str]) -> float:
        if not a or not b:
            return 0.0
        joint = self.marginal(full, a + b)
        ka = int(np.prod([self.card[v] for v in a]))
        return _mi2(joint.reshape(ka, -1))

    def lagrangian(self, beta: float) -> float:
        full = self.full_joint()
        comp = sum(self._mi(full, [t], parents) for t, (parents, _) in self.compress.items())
        rel = sum(self._mi(full, [v], list(self.out.predecessors(v))) for v in self.names)
        return comp - beta * rel

    def distortion(self, t: str) -> np.ndarray:
        """``d(t, u)`` of Theorem 7.1, shape ``(|U|, |T|)``."""
        full = self.full_joint()
        parents = self.compress[t][0]
        k = self.card[t]
        size_u = int(np.prod([self.card[p] for p in parents]))
        d = np.zeros((size_u, k))
        for child in self.out.successors(t):
            others = [v for v in self.out.predecessors(child) if v != t]
            d += self._child_term(full, t, parents, child, others)
        pa_t = list(self.out.predecessors(t))
        if pa_t:
            d += self._parent_term(full, t, parents, pa_t)
        return d

    def _child_term(self, full, t, parents, child, others) -> np.ndarray:
        # E_{P(others | u)} D(P(child | others, u) || P(child | others, t))
        u_vars = list(parents)
        if child in u_vars or set(others) & set(u_vars):
            raise ValueError("compression inputs must be disjoint from the targets' parents")
        keep = _unique(u_vars + [t] + [child] + others)
        m = self.marginal(full, keep)
        shape_u = [self.card[v] for v in u_vars]
        size_u = int(np.prod(shape_u))
        kt = self.card[t]
        out = np.zeros((size_u, kt))
        # P(u, child, others) and P(t, child, others)
        m_u = _sum_over(m, keep, [t])
        m_t = _sum_over(m, keep, u_vars)
        keep_u = [v for v in keep if v != t]
        keep_t = [v for v in keep if v not in u_vars]
        for flat_u, u in enumerate(itertools.product(*[range(c) for c in shape_u])):
            sel_u = _select(m_u, keep_u, dict(zip(u_vars, u)))  # over (child, others) or with u
            rem_u = [v for v in keep_u if v not in u_vars]
            pu = sel_u.sum()
            if pu <= 0:
                continue
            cond_u = _conditional(sel_u, rem_u, child)  # P(child | others, u), P(others | u)
            for tv in range(kt):
                sel_t = _select(m_t, keep_t, {t: tv})
                rem_t = [v for v in keep_t if v != t]
                if sel_t.sum() <= 0:
                    out[flat_u, tv] = np.inf
                    continue
                cond_t = _conditional(_reorder(sel_t, rem_t, rem_u), rem_u, child)
                out[flat_u, tv] = _expected_kl(cond_u, cond_t)
        return out

    def _parent_term(self, full, t, parents, pa_t) -> np.ndarray:
        # D(P(pa_t | u) || P(pa_t | t))
        u_vars = list(parents)
        size_u = int(np.prod([self.card[v] for v in u_vars]))
        kt = self.card[t]
        a = self.marginal(full, u_vars + pa_t).reshape(size_u, -1)
        b = self.marginal(full, [t] + pa_t).reshape(kt, -1)
        a = a / np.maximum(a.sum(axis=1, keepdims=True), _FLOOR)
        b = b / np.maximum(b.sum(axis=1, keepdims=True), _FLOOR)
        return kl_rows(a, b)

    def step(self, t: str, beta: float) -> None:
        """One asynchronous update of ``P(t | u)`` by Theorem 7.1."""
        full = self.full_joint()
        p_t = self.marginal(full, [t])
        d = self.distortion(t)
        logits = np.log(np.maximum(p_t, _FLOOR))[None, :] - beta * d
        logits -= logits.max(axis=1, keepdims=True)
        w = np.exp(logits)
        w[:, p_t <= 0] = 0.0
        self.encoders[t] = w / w.sum(axis=1, keepdims=True)


def _unique(seq):
    out = []
    for v in seq:
        if v not in out:
            out.append(v)
    return out


def _sum_over(m, names, drop):
    axes = tuple(i for i, v in enumerate(names) if v in drop)
    return m.sum(axis=axes) if axes else m


def _select(m, names, fix: dict):
    idx = tuple(fix[v] if v in fix else slice(None) for v in names)
    return m[idx]


def _reorder(arr, names_from, names_to):
    perm = [names_from.index(v) for v in names_to]
    return np.transpose(arr, perm)


def _conditional(arr, names, child):
    """Split ``arr`` over ``names`` into P(child | rest) and P(rest)."""
    ci = names.index(child)
    arr = np.moveaxis(arr, ci, 0)
    total = arr.sum()
    rest = arr.sum(axis=0)
    cond = arr / np.maximum(rest[None, ...], _FLOOR)
    return cond, rest / total


def _expected_kl(cond_u, cond_t) -> float:
    pc_u, prest = cond_u
    pc_t, _ = cond_t
    p = pc_u.reshape(pc_u.shape[0], -1)
    q = pc_t.reshape(pc_t.shape[0], -1)
    w = prest.ravel()
    total = 0.0
    for r in range(p.shape[1]):
        if w[r] <= 0:
            continue
        pr, qr = p[:, r], q[:, r]
        s = pr > 0
        if (qr[s] <= 0).any():
            return float("inf")
        total += w[r] * float(np.sum(pr[s] * (np.log(pr[s]) - np.log(qr[s]))))
    return total


def _mi2(joint: np.ndarray) -> float:
    pa = joint.sum(axis=1)
    pb = joint.sum(axis=0)
    return float(entropy(pa) + entropy(pb) - entropy(joint))


# ---------------------------------------------------------------------------
# Reach-practice specification at survey scale


@dataclass
class RPRSolution:
    beta: float
    enc_a: np.ndarray  # P(t_A | a), rows over training persons as values of A
    enc_m: np.ndarray  # P(t_M | m)
    person_a: np.ndarray  # P(t_A | person)
    person_m: np.ndarray  # P(t_M | person)
    lagrangian: float
    compression_a: float
    compression_m: float
    relevance_link: float  # I(T_A; T_M)
    relevance_v: float  # I(V; T_A, T_M)
    n_iter: int
    converged: bool
    history: list[float] = field(default_factory=list)

    @property
    def k_a(self) -> int:
        return int((self.person_a.mean(axis=0) > 1e-9).sum())

    @property
    def k_m(self) -> int:
        return int((self.person_m.mean(axis=0) > 1e-9).sum())

    def strata(self) -> np.ndarray:
        """Joint stratifier memberships ``P(t_A, t_M | person)``, shape (n, k_A * k_M)."""
        return (self.person_a[:, :, None] * self.person_m[:, None, :]).reshape(
            self.person_a.shape[0], -1)


class ReachPracticeBottleneck:
    """The reach-practice specification on persons with smoothers ``W_A`` and ``W_M``.

    Parameters
    ----------
    relevance
        ``P(v | person)``, shape ``(n, |V|)``.
    w_a, w_m
        Row-stochastic smoothers, shape ``(n, n)``: ``W[i, a]`` is the probability that
        person ``i`` is represented by training person ``a``'s configuration.
    p_person
        Person weights; uniform by default.
    """

    def __init__(self, relevance: np.ndarray, w_a: np.ndarray, w_m: np.ndarray,
                 p_person: np.ndarray | None = None) -> None:
        from scipy import sparse

        self.q = np.asarray(relevance, dtype=float)
        n = self.q.shape[0]
        self.w_a = sparse.csr_matrix(np.asarray(w_a, dtype=float))
        self.w_m = sparse.csr_matrix(np.asarray(w_m, dtype=float))
        self.p_i = np.full(n, 1.0 / n) if p_person is None else np.asarray(p_person, float)
        self.p_a = np.asarray(self.w_a.T @ self.p_i).ravel()
        self.p_m = np.asarray(self.w_m.T @ self.p_i).ravel()
        # P(i | a) and P(i | m), rows over values of A and M
        da = sparse.diags(1.0 / np.maximum(self.p_a, _FLOOR))
        dm = sparse.diags(1.0 / np.maximum(self.p_m, _FLOOR))
        di = sparse.diags(self.p_i)
        self.qa = (da @ (di @ self.w_a).T).tocsr()
        self.qm = (dm @ (di @ self.w_m).T).tocsr()

    def lagrangian(self, enc_a, enc_m, beta):
        jj = self._joint(self.w_a @ enc_a, self.w_m @ enc_m)
        comp_a = _mi2(self.p_a[:, None] * enc_a)
        comp_m = _mi2(self.p_m[:, None] * enc_m)
        link = _mi2(jj.sum(axis=2))
        rel = _mi2(jj.reshape(-1, jj.shape[2]))
        return comp_a + comp_m - beta * (link + rel), comp_a, comp_m, link, rel

    def _joint(self, ga, gm):
        return np.einsum("i,it,is,iv->tsv", self.p_i, ga, gm, self.q)

    def _smoothed(self, mat, g):
        """``sum_i mat[a, i] g[i, t] q[i, v]`` for every (a, t, v)."""
        return np.stack([np.asarray(mat @ (g[:, [t]] * self.q)) for t in range(g.shape[1])],
                        axis=1)

    def distortion_a(self, enc_a, enc_m):
        ga, gm = self.w_a @ enc_a, self.w_m @ enc_m
        ta = self._smoothed(self.qa, gm)
        jj = self._joint(ga, gm)
        p_tm_a = ta.sum(axis=2)  # P(t_M | a)
        p_tm_ta = jj.sum(axis=2)
        p_tm_ta = p_tm_ta / np.maximum(p_tm_ta.sum(axis=1, keepdims=True), _FLOOR)
        d = kl_rows(_norm(p_tm_a), p_tm_ta)
        v_ta_tm = jj / np.maximum(jj.sum(axis=2, keepdims=True), _FLOOR)  # P(v | t_A, t_M)
        for s in range(enc_m.shape[1]):
            w = p_tm_a[:, s]
            if (w <= 0).all():
                continue
            v_a = _norm(ta[:, s, :])  # P(v | t_M = s, a)
            d += w[:, None] * kl_rows(v_a, v_ta_tm[:, s, :])
        return d

    def distortion_m(self, enc_a, enc_m):
        ga, gm = self.w_a @ enc_a, self.w_m @ enc_m
        tm = self._smoothed(self.qm, ga)
        jj = self._joint(ga, gm)
        p_ta_m = tm.sum(axis=2)  # P(t_A | m)
        p_ta_tm = jj.sum(axis=2).T
        p_ta_tm = p_ta_tm / np.maximum(p_ta_tm.sum(axis=1, keepdims=True), _FLOOR)
        d = kl_rows(_norm(p_ta_m), p_ta_tm)
        v_ta_tm = jj / np.maximum(jj.sum(axis=2, keepdims=True), _FLOOR)
        for s in range(enc_a.shape[1]):
            w = p_ta_m[:, s]
            if (w <= 0).all():
                continue
            v_m = _norm(tm[:, s, :])  # P(v | t_A = s, m)
            d += w[:, None] * kl_rows(v_m, v_ta_tm[s, :, :])
        return d

    # iteration ---------------------------------------------------------------
    def iterate(self, enc_a, enc_m, beta, max_iter=300, tol=1e-8,
                record: bool = False) -> RPRSolution:
        enc_a = np.asarray(enc_a, float).copy()
        enc_m = np.asarray(enc_m, float).copy()
        history = []
        converged = False
        it = 0
        for it in range(1, max_iter + 1):
            if record:
                history.append(self.lagrangian(enc_a, enc_m, beta)[0])
            new_a = _update(self.p_a, enc_a, self.distortion_a(enc_a, enc_m), beta)
            new_m = _update(self.p_m, enc_m, self.distortion_m(new_a, enc_m), beta)
            change = max(np.abs(new_a - enc_a).max(), np.abs(new_m - enc_m).max())
            enc_a, enc_m = new_a, new_m
            if change < tol:
                converged = True
                break
        lag, ca, cm, link, rel = self.lagrangian(enc_a, enc_m, beta)
        if record:
            history.append(lag)
        return RPRSolution(beta, enc_a, enc_m, self.w_a @ enc_a, self.w_m @ enc_m, lag, ca,
                           cm, link, rel, it, converged, history)

    def anneal(self, betas, k_a_max=4, k_m_max=4, perturbation=0.05, merge_tol=1e-4,
               seed=None, max_iter=300) -> list[RPRSolution]:
        """Deterministic annealing with duplication of both regime variables."""
        rng = np.random.default_rng(seed)
        n = self.q.shape[0]
        enc_a = np.ones((n, 1))
        enc_m = np.ones((n, 1))
        path = []
        for beta in np.asarray(betas, float):
            enc_a = _duplicate(enc_a, k_a_max, perturbation, rng)
            enc_m = _duplicate(enc_m, k_m_max, perturbation, rng)
            sol = self.iterate(enc_a, enc_m, beta, max_iter=max_iter)
            enc_a = _merge(sol.enc_a, self.p_a, merge_tol)
            enc_m = _merge(sol.enc_m, self.p_m, merge_tol)
            sol = self.iterate(enc_a, enc_m, beta, max_iter=max_iter)
            enc_a = sol.enc_a[:, (self.p_a @ sol.enc_a) > 1e-12]
            enc_m = sol.enc_m[:, (self.p_m @ sol.enc_m) > 1e-12]
            path.append(sol)
        return path


def _norm(m):
    return m / np.maximum(m.sum(axis=-1, keepdims=True), _FLOOR)


def _update(p_u, enc, d, beta):
    p_t = p_u @ enc
    logits = np.log(np.maximum(p_t, _FLOOR))[None, :] - beta * d
    logits -= logits.max(axis=1, keepdims=True)
    w = np.exp(logits)
    w[:, p_t <= 0] = 0.0
    return w / w.sum(axis=1, keepdims=True)


def _duplicate(enc, k_max, perturbation, rng):
    if enc.shape[1] >= k_max:
        return enc
    eps = rng.uniform(-0.5, 0.5, size=enc.shape)
    dup = np.concatenate([enc * (0.5 + perturbation * eps), enc * (0.5 - perturbation * eps)],
                         axis=1)[:, :k_max]
    return dup / dup.sum(axis=1, keepdims=True)


def _merge(enc, p_u, tol):
    keep = (p_u @ enc) > 1e-12
    enc = enc[:, keep]
    groups: list[list[int]] = []
    for t in range(enc.shape[1]):
        for g in groups:
            if np.abs(enc[:, g[0]] / max(enc[:, g[0]].sum(), _FLOOR)
                      - enc[:, t] / max(enc[:, t].sum(), _FLOOR)).max() < tol:
                g.append(t)
                break
        else:
            groups.append([t])
    return np.stack([enc[:, g].sum(axis=1) for g in groups], axis=1)
