"""Entropy, divergence and mutual information for discrete distributions (natural logarithm).

Every function takes probability arrays and returns values in nats; divide by
``log(2)`` for bits.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-300


def entropy(p: np.ndarray, axis: int | None = None) -> np.ndarray | float:
    """Shannon entropy ``-sum p log p``."""
    p = np.asarray(p, dtype=float)
    terms = np.where(p > 0, -p * np.log(np.where(p > 0, p, 1.0)), 0.0)
    return terms.sum(axis=axis) if axis is not None else float(terms.sum())


def kl(p: np.ndarray, q: np.ndarray) -> float:
    """Kullback-Leibler divergence ``D(p || q)``; infinite when p has mass where q has none."""
    p = np.asarray(p, dtype=float).ravel()
    q = np.asarray(q, dtype=float).ravel()
    support = p > 0
    if (q[support] <= 0).any():
        return float("inf")
    return float(np.sum(p[support] * (np.log(p[support]) - np.log(q[support]))))


def mutual_information(joint: np.ndarray) -> float:
    """``I(A; B)`` for a two-dimensional joint array ``P(a, b)``."""
    joint = np.asarray(joint, dtype=float)
    joint = joint / joint.sum()
    pa = joint.sum(axis=1, keepdims=True)
    pb = joint.sum(axis=0, keepdims=True)
    return kl(joint, pa * pb)


def conditional_mutual_information(
    p_c: np.ndarray, q_c: np.ndarray, p_t_given_c: np.ndarray
) -> float:
    """``I(C; V | T)`` when T is a stochastic function of C (Markov chain T - C - V).

    Parameters
    ----------
    p_c
        Distribution of C, shape ``(nc,)``.
    q_c
        Conditional distributions ``P(V | c)``, shape ``(nc, nv)``.
    p_t_given_c
        Encoder ``P(t | c)``, shape ``(nc, nt)``.
    """
    p_c = np.asarray(p_c, dtype=float)
    q_c = np.asarray(q_c, dtype=float)
    enc = np.asarray(p_t_given_c, dtype=float)
    p_ct = p_c[:, None] * enc
    p_t = p_ct.sum(axis=0)
    total = 0.0
    for t in range(enc.shape[1]):
        if p_t[t] <= 0:
            continue
        w = p_ct[:, t] / p_t[t]
        q_t = w @ q_c
        for c in range(q_c.shape[0]):
            if p_ct[c, t] > 0:
                total += p_ct[c, t] * kl(q_c[c], q_t)
    return float(total)


def relevant_information(p_c: np.ndarray, q_c: np.ndarray, p_t_given_c: np.ndarray) -> float:
    """``I(T; V)`` for T a stochastic function of C with conditionals ``P(V | c)``."""
    p_ct = np.asarray(p_c, dtype=float)[:, None] * np.asarray(p_t_given_c, dtype=float)
    joint_tv = p_ct.T @ np.asarray(q_c, dtype=float)
    return mutual_information(joint_tv)


def compression_information(p_c: np.ndarray, p_t_given_c: np.ndarray) -> float:
    """``I(C; T)`` for an encoder ``P(t | c)``."""
    p_ct = np.asarray(p_c, dtype=float)[:, None] * np.asarray(p_t_given_c, dtype=float)
    return mutual_information(p_ct)
