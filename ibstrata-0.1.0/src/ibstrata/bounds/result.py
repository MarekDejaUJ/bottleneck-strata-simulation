"""Result type shared by every bound computation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Bounds:
    """An identification interval for one probability of causation.

    ``method`` names the source of the interval: ``"exact LP"`` (sharp under the
    margins alone), ``"Tian-Pearl"``, ``"Li-Pearl Theorem k"``, ``"Shu-Wang-Li Theorem 2"``,
    ``"monotonicity"`` (a point) or ``"stratified ..."``. Witnesses, when present, are
    joint distributions of the choice and the potential outcomes, flattened in the
    order of :func:`ibstrata.bounds.margins.response_cells`, that attain the endpoints.
    """

    lower: float
    upper: float
    method: str
    lower_witness: np.ndarray | None = None
    upper_witness: np.ndarray | None = None

    @property
    def width(self) -> float:
        return self.upper - self.lower

    def contains(self, value: float, tol: float = 1e-9) -> bool:
        return self.lower - tol <= value <= self.upper + tol

    def __iter__(self):
        yield self.lower
        yield self.upper

    def __repr__(self) -> str:
        return f"Bounds([{self.lower:.4f}, {self.upper:.4f}], method={self.method!r})"
