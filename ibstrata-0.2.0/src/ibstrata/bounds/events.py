"""Events on the choice of treatment and the potential outcomes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType


@dataclass(frozen=True)
class Event:
    """A conjunction of conditions on potential outcomes and on observed variables.

    ``do`` maps a treatment index ``j`` to an outcome index ``i`` and states
    ``Y_{x_j} = y_i``. ``x`` states that the unit's own choice is ``x``; ``y`` states
    that the observed outcome, ``Y_{X}``, equals ``y``.

    Examples
    --------
    PNS for binary treatment and outcome with ``x = 1``, ``y = 1``:
    ``Event(do={1: 1, 0: 0})``. The numerator of PN: ``Event(do={0: 0}, x=1, y=1)``.
    """

    do: Mapping[int, int] = field(default_factory=dict)
    x: int | None = None
    y: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "do", MappingProxyType(dict(self.do)))

    def holds(self, x: int, ys: tuple[int, ...]) -> bool:
        """Whether the event holds in the cell ``(x, ys)`` of the response joint."""
        for j, i in self.do.items():
            if ys[j] != i:
                return False
        if self.x is not None and x != self.x:
            return False
        if self.y is not None and ys[x] != self.y:
            return False
        return True

    def observed_only(self) -> bool:
        """True when the event involves no potential outcome."""
        return len(self.do) == 0

    def conjoin(self, other: Event) -> Event:
        """The conjunction of two events; conflicting conditions raise ``ValueError``."""
        do = dict(self.do)
        for j, i in other.do.items():
            if j in do and do[j] != i:
                raise ValueError("the events contradict each other")
            do[j] = i
        x = _merge(self.x, other.x)
        y = _merge(self.y, other.y)
        return Event(do=do, x=x, y=y)

    def describe(self, x_labels: tuple[str, ...], y_labels: tuple[str, ...]) -> str:
        """Readable form such as ``P(y2_{x1}, y1_{x2}, x1, y2)``."""
        parts = [f"{y_labels[i]}_{{{x_labels[j]}}}" for j, i in sorted(self.do.items())]
        if self.x is not None:
            parts.append(x_labels[self.x])
        if self.y is not None:
            parts.append(y_labels[self.y])
        return "P(" + ", ".join(parts) + ")"


def _merge(a: int | None, b: int | None) -> int | None:
    if a is None:
        return b
    if b is None or a == b:
        return a
    raise ValueError("the events contradict each other")
