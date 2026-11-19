"""Multinomial logistic regression for conditional distributions of a categorical target.

The relevance model of the bottleneck needs ``P(X, Y | S = s, C = c)`` for every design
arm ``s``. ``fit_softmax`` estimates one such conditional distribution by penalized
maximum likelihood; :class:`ibstrata.stratify.LogisticRelevance` assembles the arms.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp


@dataclass
class SoftmaxModel:
    """Fitted multinomial logistic regression on standardized features."""

    coef: np.ndarray  # (n_classes, d)
    intercept: np.ndarray  # (n_classes,)
    center: np.ndarray  # (d,)
    scale: np.ndarray  # (d,)

    def predict(self, features: np.ndarray) -> np.ndarray:
        """``P(class | features)``, one row per row of ``features``."""
        z = (np.asarray(features, dtype=float) - self.center) / self.scale
        logits = z @ self.coef.T + self.intercept
        return np.exp(logits - logsumexp(logits, axis=1, keepdims=True))


def fit_softmax(features: np.ndarray, target: np.ndarray, n_classes: int,
                penalty: float = 1.0, pseudo: float = 0.05,
                max_iter: int = 500) -> SoftmaxModel:
    """Multinomial logistic regression by penalized maximum likelihood.

    The objective is the negative log-likelihood of ``target`` plus ``penalty / 2`` times
    the squared norm of the coefficients of the standardized features, plus ``pseudo``
    observations of every class at the mean of the features. The pseudo-observations keep
    the probability of a class absent from the sample positive. Constant features are
    centered and left unscaled.
    """
    x = np.asarray(features, dtype=float)
    t = np.asarray(target, dtype=int)
    if x.ndim != 2 or t.shape != (x.shape[0],):
        raise ValueError("features must be a matrix with one row per target value")
    if x.shape[0] == 0:
        raise ValueError("a relevance model needs at least one unit")
    if t.min() < 0 or t.max() >= n_classes:
        raise ValueError("target values must lie between 0 and n_classes - 1")
    center = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    z = (x - center) / scale
    n, d = z.shape
    onehot = np.eye(n_classes)[t]
    counts = onehot.sum(axis=0)

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        coef = theta[: n_classes * d].reshape(n_classes, d)
        icpt = theta[n_classes * d:]
        logits = z @ coef.T + icpt
        logp = logits - logsumexp(logits, axis=1, keepdims=True)
        logp0 = icpt - logsumexp(icpt)
        value = (-(onehot * logp).sum() - pseudo * logp0.sum()
                 + 0.5 * penalty * (coef ** 2).sum())
        resid = np.exp(logp) - onehot
        g_coef = resid.T @ z + penalty * coef
        g_icpt = resid.sum(axis=0) + pseudo * (n_classes * np.exp(logp0) - 1.0)
        return float(value), np.concatenate([g_coef.ravel(), g_icpt])

    start = np.zeros(n_classes * d + n_classes)
    start[n_classes * d:] = np.log((counts + pseudo) / (n + pseudo * n_classes))
    res = minimize(objective, start, jac=True, method="L-BFGS-B",
                   options={"maxiter": max_iter})
    coef = res.x[: n_classes * d].reshape(n_classes, d)
    return SoftmaxModel(coef, res.x[n_classes * d:], center, scale)
