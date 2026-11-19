import numpy as np

from ibstrata.bounds import Margins


def random_joint(
    rng: np.random.Generator, m: int, n: int, concentration: float = 0.5
) -> np.ndarray:
    """A random joint of the choice and the potential outcomes, shape (m, n, ..., n)."""
    size = m * n**m
    return rng.dirichlet(np.full(size, concentration)).reshape((m,) + (n,) * m)


def margins_and_joint(rng: np.random.Generator, m: int, n: int, concentration: float = 0.5):
    joint = random_joint(rng, m, n, concentration)
    return Margins.from_joint(joint), joint
