"""Initialization strategies for the weights of a layer."""

from collections.abc import Callable

import numpy as np

from my_torch.errors import MyTorchError

RANDOM_SCALE = 0.05


def uniform(shape: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(-RANDOM_SCALE, RANDOM_SCALE, size=shape)


def normal(shape: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    return rng.normal(0.0, RANDOM_SCALE, size=shape)


def lecun(shape: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    return rng.normal(0.0, np.sqrt(1.0 / shape[0]), size=shape)


def xavier_uniform(shape: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    limit = np.sqrt(6.0 / (shape[0] + shape[1]))
    return rng.uniform(-limit, limit, size=shape)


def xavier_normal(shape: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    return rng.normal(0.0, np.sqrt(2.0 / (shape[0] + shape[1])), size=shape)


def he(shape: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    return rng.normal(0.0, np.sqrt(2.0 / shape[0]), size=shape)


INITIALIZERS = {
    "uniform": uniform,
    "normal": normal,
    "lecun": lecun,
    "xavier_uniform": xavier_uniform,
    "xavier_normal": xavier_normal,
    "he": he,
}


def get_initializer(name: str) -> Callable[[tuple[int, int], np.random.Generator], np.ndarray]:
    """Return an initialization function from its name."""
    if name not in INITIALIZERS:
        raise MyTorchError(
            f"unknown initializer '{name}', expected one of: {', '.join(INITIALIZERS)}"
        )
    return INITIALIZERS[name]
