"""Initialization strategies for the weights of a layer."""

import numpy as np

RANDOM_SCALE = 0.05


def uniform(shape: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(-RANDOM_SCALE, RANDOM_SCALE, size=shape)


def normal(shape: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    return rng.normal(0.0, RANDOM_SCALE, size=shape)
