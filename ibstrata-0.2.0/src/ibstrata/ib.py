"""The information bottleneck on items with conditional distributions of a relevance variable.

Items are the values of the compressed variable C (persons, covariate configurations);
each item has a probability ``p_c`` and a conditional distribution ``P(V | c)`` of the
relevance variable. A solution is an encoder ``P(t | c)`` that minimizes
``I(C; T) - beta I(T; V)`` (Tishby, Pereira and Bialek, 1999). Three algorithms:

* iterative self-consistent equations at fixed beta, with deterministic annealing over
  beta (cluster duplication and perturbation, merging of copies that do not separate);
* agglomerative bottleneck (Slonim and Tishby, 2000): greedy hard merges that lose the
  least relevant information;
* deterministic bottleneck (Strouse and Schwab, 2017): hard encoder minimizing
  ``H(T) - beta I(T; V)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .info import entropy

_FLOOR = 1e-300


def kl_rows(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Matrix of ``D(p_i || q_t)`` for rows ``p_i`` of ``p`` and rows ``q_t`` of ``q``."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    plogp = np.where(p > 0, p * np.log(np.where(p > 0, p, 1.0)), 0.0).sum(axis=1)
    logq = np.log(np.maximum(q, _FLOOR))
    cross = p @ logq.T
    return plogp[:, None] - cross


@dataclass
class IBSolution:
    """One solution of the bottleneck at a given beta."""

    beta: float
    encoder: np.ndarray
    p_t: np.ndarray
    decoder: np.ndarray
    compression: float
    relevance: float
    n_iter: int
    converged: bool
    history: list[float] = field(default_factory=list)

    @property
    def k(self) -> int:
        return int((self.p_t > 1e-12).sum())

    @property
    def lagrangian(self) -> float:
        return self.compression - self.beta * self.relevance

    def hard(self) -> np.ndarray:
        """Most probable stratum of every item."""
        return self.encoder.argmax(axis=1)


def _summaries(p_c: np.ndarray, q: np.ndarray, enc: np.ndarray):
    p_ct = p_c[:, None] * enc
    p_t = p_ct.sum(axis=0)
    safe = np.where(p_t > 0, p_t, 1.0)
    dec = (p_ct.T @ q) / safe[:, None]
    p_v = p_c @ q
    comp = _mi_from_joint(p_ct)
    rel = _mi_from_joint(p_t[:, None] * dec)
    return p_t, dec, comp, rel, p_v


def _mi_from_joint(joint: np.ndarray) -> float:
    pa = joint.sum(axis=1)
    pb = joint.sum(axis=0)
    return float(entropy(pa) + entropy(pb) - entropy(joint))


def _check(p_c, q):
    p_c = np.asarray(p_c, dtype=float)
    q = np.asarray(q, dtype=float)
    if q.ndim != 2 or q.shape[0] != p_c.shape[0]:
        raise ValueError("q must have one row of P(V | c) per item")
    if not np.isclose(p_c.sum(), 1.0) or (p_c < 0).any():
        raise ValueError("p_c must be a probability vector")
    if not np.allclose(q.sum(axis=1), 1.0, atol=1e-8) or (q < 0).any():
        raise ValueError("every row of q must be a probability vector")
    return p_c, q


def iterate(
    p_c: np.ndarray,
    q: np.ndarray,
    encoder: np.ndarray,
    beta: float,
    max_iter: int = 2000,
    tol: float = 1e-11,
) -> IBSolution:
    """Iterate the self-consistent equations from ``encoder`` at fixed ``beta``.

    ``P(t | c) = P(t) exp(-beta D[P(V | c) || P(V | t)]) / Z(c, beta)``. The Lagrangian
    of every iterate is kept in ``history``.
    """
    p_c, q = _check(p_c, q)
    enc = np.asarray(encoder, dtype=float).copy()
    history: list[float] = []
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        p_t, dec, comp, rel, _ = _summaries(p_c, q, enc)
        history.append(comp - beta * rel)
        d = kl_rows(q, dec)
        logits = np.log(np.maximum(p_t, _FLOOR))[None, :] - beta * d
        logits -= logits.max(axis=1, keepdims=True)
        new = np.exp(logits)
        new[:, p_t <= 0] = 0.0
        new /= new.sum(axis=1, keepdims=True)
        change = float(np.abs(new - enc).max())
        enc = new
        if change < tol:
            converged = True
            break
    p_t, dec, comp, rel, _ = _summaries(p_c, q, enc)
    history.append(comp - beta * rel)
    return IBSolution(beta, enc, p_t, dec, comp, rel, it, converged, history)


def _merge_duplicates(sol: IBSolution, p_c, q, merge_tol: float) -> np.ndarray:
    """Merge strata whose decoders coincide; drop empty strata."""
    enc = sol.encoder[:, sol.p_t > 1e-12]
    dec = sol.decoder[sol.p_t > 1e-12]
    groups: list[list[int]] = []
    for t in range(dec.shape[0]):
        for g in groups:
            if np.abs(dec[g[0]] - dec[t]).max() < merge_tol:
                g.append(t)
                break
        else:
            groups.append([t])
    return np.stack([enc[:, g].sum(axis=1) for g in groups], axis=1)


@dataclass
class AnnealingPath:
    """Solutions along an increasing beta schedule (the information curve)."""

    solutions: list[IBSolution]

    @property
    def betas(self) -> np.ndarray:
        return np.array([s.beta for s in self.solutions])

    @property
    def curve(self) -> np.ndarray:
        """Rows ``(beta, I(C;T), I(T;V), k)``."""
        return np.array([[s.beta, s.compression, s.relevance, s.k] for s in self.solutions])

    def at_k(self, k: int) -> IBSolution | None:
        """The first solution on the path with ``k`` strata."""
        for s in self.solutions:
            if s.k == k:
                return s
        return None

    def bifurcations(self) -> list[tuple[float, int]]:
        """Betas at which the number of strata grows, with the new count."""
        out = []
        prev = 1
        for s in self.solutions:
            if s.k > prev:
                out.append((s.beta, s.k))
                prev = s.k
        return out


def anneal(
    p_c: np.ndarray,
    q: np.ndarray,
    betas: np.ndarray,
    k_max: int = 8,
    perturbation: float = 0.05,
    merge_tol: float = 1e-4,
    seed: int | None = None,
    max_iter: int = 2000,
) -> AnnealingPath:
    """Deterministic annealing over an increasing beta schedule.

    At every beta each stratum is duplicated, the copies are perturbed by
    ``P(t | c) (1/2 +/- alpha eps(c))`` with ``eps`` uniform on [-1/2, 1/2] (Friedman et al.,
    2001, Section 8), the self-consistent equations are iterated, and copies whose
    decoders coincide are merged back. The count of strata never exceeds ``k_max``.
    """
    p_c, q = _check(p_c, q)
    rng = np.random.default_rng(seed)
    enc = np.ones((p_c.shape[0], 1))
    sols: list[IBSolution] = []
    for beta in np.asarray(betas, dtype=float):
        k_now = enc.shape[1]
        if k_now < k_max:
            eps = rng.uniform(-0.5, 0.5, size=enc.shape)
            dup = np.concatenate([enc * (0.5 + perturbation * eps),
                                  enc * (0.5 - perturbation * eps)], axis=1)
            if dup.shape[1] > k_max:
                dup = dup[:, : k_max]
            dup /= dup.sum(axis=1, keepdims=True)
        else:
            dup = enc
        sol = iterate(p_c, q, dup, beta, max_iter=max_iter)
        enc = _merge_duplicates(sol, p_c, q, merge_tol)
        sol = iterate(p_c, q, enc, beta, max_iter=max_iter)
        enc = sol.encoder[:, sol.p_t > 1e-12]
        sols.append(sol)
    return AnnealingPath(sols)


@dataclass
class AgglomerativeTree:
    """Hard merges of the agglomerative bottleneck, from n items to one stratum."""

    merges: list[tuple[int, int, float]]  # (cluster a, cluster b, loss of I(T;V))
    labels_by_k: dict[int, np.ndarray]
    relevance_by_k: dict[int, float]


def _row_entropy(m: np.ndarray) -> np.ndarray:
    return -np.where(m > 0, m * np.log(np.where(m > 0, m, 1.0)), 0.0).sum(axis=-1)


def agglomerative(p_c: np.ndarray, q: np.ndarray, k_min: int = 1) -> AgglomerativeTree:
    """Agglomerative bottleneck (Slonim and Tishby, 2000).

    Merging strata a and b loses ``(p_a + p_b) JS_w(P(V | a), P(V | b))`` of relevant
    information, with weights ``w = (p_a, p_b) / (p_a + p_b)``; the pair with the
    smallest loss is merged at every step.
    """
    p_c, q = _check(p_c, q)
    n = p_c.shape[0]
    mass = p_c.copy()
    cond = q.copy()
    ent = _row_entropy(cond)
    alive = np.ones(n, dtype=bool)
    ids = list(range(n))
    members = {i: [i] for i in range(n)}

    def costs_against(a: int) -> np.ndarray:
        tot = mass[a] + mass
        safe = np.where(tot > 0, tot, 1.0)
        mix = (mass[a] * cond[a][None, :] + mass[:, None] * cond) / safe[:, None]
        c = tot * _row_entropy(mix) - mass[a] * ent[a] - mass * ent
        c[~alive] = np.inf
        c[a] = np.inf
        return np.maximum(c, 0.0)

    cost = np.full((n, n), np.inf)
    for a in range(n):
        cost[a] = costs_against(a)
    merges: list[tuple[int, int, float]] = []
    labels_by_k: dict[int, np.ndarray] = {}
    relevance_by_k: dict[int, float] = {}
    nxt = n

    def snapshot():
        act = np.flatnonzero(alive)
        lab = np.empty(n, dtype=int)
        for idx, a in enumerate(act):
            lab[members[ids[a]]] = idx
        labels_by_k[len(act)] = lab
        relevance_by_k[len(act)] = _mi_from_joint(mass[act][:, None] * cond[act])

    snapshot()
    while alive.sum() > k_min:
        flat = int(np.argmin(cost))
        a, b = divmod(flat, n)
        loss = float(cost[a, b])
        new_mass = mass[a] + mass[b]
        if new_mass > 0:
            cond[a] = (mass[a] * cond[a] + mass[b] * cond[b]) / new_mass
        mass[a] = new_mass
        ent[a] = _row_entropy(cond[a][None, :])[0]
        alive[b] = False
        mass[b] = 0.0
        merges.append((ids[a], ids[b], loss))
        members[nxt] = members.pop(ids[a]) + members.pop(ids[b])
        ids[a] = nxt
        nxt += 1
        cost[b, :] = np.inf
        cost[:, b] = np.inf
        row = costs_against(a)
        cost[a, :] = row
        cost[:, a] = row
        snapshot()
    return AgglomerativeTree(merges, labels_by_k, relevance_by_k)


def deterministic(
    p_c: np.ndarray,
    q: np.ndarray,
    k: int,
    beta: float,
    seed: int | None = None,
    max_iter: int = 500,
) -> IBSolution:
    """Deterministic bottleneck (Strouse and Schwab, 2017): hard assignment
    ``t(c) = argmax_t [log P(t) - beta D(P(V | c) || P(V | t))]``."""
    p_c, q = _check(p_c, q)
    rng = np.random.default_rng(seed)
    lab = rng.integers(k, size=p_c.shape[0])
    it = 0
    for it in range(1, max_iter + 1):
        enc = np.eye(k)[lab]
        p_t, dec, _, _, _ = _summaries(p_c, q, enc)
        d = kl_rows(q, dec)
        score = np.log(np.maximum(p_t, _FLOOR))[None, :] - beta * d
        score[:, p_t <= 0] = -np.inf
        new = score.argmax(axis=1)
        if (new == lab).all():
            break
        lab = new
    enc = np.eye(k)[lab]
    p_t, dec, comp, rel, _ = _summaries(p_c, q, enc)
    return IBSolution(beta, enc, p_t, dec, comp, rel, it, True, [])


def encode(solution: IBSolution, q_new: np.ndarray) -> np.ndarray:
    """Memberships ``P(t | c)`` of new items from their conditionals ``P(V | c)``, with
    the fitted ``P(t)`` and ``P(V | t)`` held fixed (one self-consistent step)."""
    q_new = np.atleast_2d(np.asarray(q_new, dtype=float))
    keep = solution.p_t > 1e-12
    d = kl_rows(q_new, solution.decoder[keep])
    logits = np.log(solution.p_t[keep])[None, :] - solution.beta * d
    logits -= logits.max(axis=1, keepdims=True)
    w = np.exp(logits)
    out = np.zeros((q_new.shape[0], solution.p_t.shape[0]))
    out[:, keep] = w / w.sum(axis=1, keepdims=True)
    return out
