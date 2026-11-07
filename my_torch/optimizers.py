"""Optimizers updating the parameters of a network from their gradients."""

import numpy as np

from my_torch.errors import MyTorchError
from my_torch.layers import Dense


def check_positive(name: str, value: float) -> None:
    if not value > 0:
        raise MyTorchError(f"{name} must be positive, got {value}")


def check_decay(name: str, value: float) -> None:
    if not 0 <= value < 1:
        raise MyTorchError(f"{name} must be in [0, 1[, got {value}")


def parse_state_key(key: str, layers: list[Dense]) -> tuple[str, int, str]:
    """Split a state key such as 'velocities/0/weights' into its three parts."""
    parts = key.split("/")
    if (
        len(parts) != 3
        or not parts[1].isdigit()
        or int(parts[1]) >= len(layers)
        or parts[2] not in ("weights", "biases")
    ):
        raise MyTorchError(f"invalid optimizer state key '{key}'")
    return parts[0], int(parts[1]), parts[2]


class Optimizer:
    """Update the parameters of the layers from their gradients."""

    def __init__(self, learning_rate: float = 0.01) -> None:
        check_positive("learning rate", learning_rate)
        self.learning_rate = learning_rate

    def step(self, layers: list[Dense]) -> None:
        """Apply one update to the weights and biases of every layer."""
        for index, layer in enumerate(layers):
            if layer.weights_gradient is None:
                raise MyTorchError("backward must be called before step")
            layer.weights -= self.compute_update((index, "weights"), layer.weights_gradient)
            layer.biases -= self.compute_update((index, "biases"), layer.biases_gradient)

    def compute_update(self, key: tuple[int, str], gradient: np.ndarray) -> np.ndarray:
        """Return the quantity to subtract from the parameter identified by key."""
        raise NotImplementedError

    def hyperparameters(self) -> dict[str, float | bool]:
        """Return the parameters needed to create the same optimizer again."""
        return {"learning_rate": float(self.learning_rate)}

    def memories(self) -> dict[str, dict[tuple[int, str], np.ndarray]]:
        """Return the arrays kept for each parameter, by kind of memory."""
        return {}

    def state(self) -> dict[str, np.ndarray]:
        """Return the memory of the optimizer as arrays named 'kind/layer/parameter'."""
        return {
            f"{kind}/{index}/{parameter}": array
            for kind, memory in self.memories().items()
            for (index, parameter), array in memory.items()
        }

    def load_state(self, state: dict[str, np.ndarray], layers: list[Dense]) -> None:
        """Replace the memory of the optimizer by a state exported for these layers."""
        memories = self.memories()
        loaded: dict[str, dict[tuple[int, str], np.ndarray]] = {kind: {} for kind in memories}
        for key, array in state.items():
            kind, index, parameter = parse_state_key(key, layers)
            if kind not in memories:
                raise MyTorchError(f"unexpected optimizer state '{key}'")
            expected_shape = getattr(layers[index], parameter).shape
            if (
                not isinstance(array, np.ndarray)
                or array.dtype.kind != "f"
                or array.shape != expected_shape
            ):
                raise MyTorchError(
                    f"optimizer state '{key}' must be a float array of shape {expected_shape}"
                )
            if not np.all(np.isfinite(array)) or (kind == "squares" and np.any(array < 0)):
                raise MyTorchError(f"optimizer state '{key}' holds invalid values")
            loaded[kind][(index, parameter)] = array.astype(np.float64)
        for kind, memory in memories.items():
            memory.clear()
            memory.update(loaded[kind])


class SGD(Optimizer):
    """Plain gradient descent."""

    def compute_update(self, key: tuple[int, str], gradient: np.ndarray) -> np.ndarray:
        return self.learning_rate * gradient


class Momentum(Optimizer):
    """Gradient descent with a velocity, and an optional Nesterov look-ahead."""

    def __init__(
        self, learning_rate: float = 0.01, momentum: float = 0.9, nesterov: bool = False
    ) -> None:
        super().__init__(learning_rate)
        check_decay("momentum", momentum)
        self.momentum = momentum
        self.nesterov = nesterov
        self.velocities: dict[tuple[int, str], np.ndarray] = {}

    def hyperparameters(self) -> dict[str, float | bool]:
        return {
            **super().hyperparameters(),
            "momentum": float(self.momentum),
            "nesterov": bool(self.nesterov),
        }

    def memories(self) -> dict[str, dict[tuple[int, str], np.ndarray]]:
        return {"velocities": self.velocities}

    def compute_update(self, key: tuple[int, str], gradient: np.ndarray) -> np.ndarray:
        velocity = self.momentum * self.velocities.get(key, 0.0) + gradient
        self.velocities[key] = velocity
        if self.nesterov:
            return self.learning_rate * (gradient + self.momentum * velocity)
        return self.learning_rate * velocity


class RMSProp(Optimizer):
    """Gradient descent dividing each step by the recent size of its gradients."""

    def __init__(
        self, learning_rate: float = 0.001, decay: float = 0.9, epsilon: float = 1e-8
    ) -> None:
        super().__init__(learning_rate)
        check_decay("decay", decay)
        check_positive("epsilon", epsilon)
        self.decay = decay
        self.epsilon = epsilon
        self.squares: dict[tuple[int, str], np.ndarray] = {}

    def hyperparameters(self) -> dict[str, float | bool]:
        return {
            **super().hyperparameters(),
            "decay": float(self.decay),
            "epsilon": float(self.epsilon),
        }

    def memories(self) -> dict[str, dict[tuple[int, str], np.ndarray]]:
        return {"squares": self.squares}

    def compute_update(self, key: tuple[int, str], gradient: np.ndarray) -> np.ndarray:
        square = self.decay * self.squares.get(key, 0.0) + (1 - self.decay) * gradient ** 2
        self.squares[key] = square
        return self.learning_rate * gradient / (np.sqrt(square) + self.epsilon)


class Adam(Optimizer):
    """Gradient descent using running means of the gradients and of their squares."""

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__(learning_rate)
        check_decay("beta1", beta1)
        check_decay("beta2", beta2)
        check_positive("epsilon", epsilon)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.steps = 0
        self.means: dict[tuple[int, str], np.ndarray] = {}
        self.squares: dict[tuple[int, str], np.ndarray] = {}

    def step(self, layers: list[Dense]) -> None:
        self.steps += 1
        super().step(layers)

    def hyperparameters(self) -> dict[str, float | bool]:
        return {
            **super().hyperparameters(),
            "beta1": float(self.beta1),
            "beta2": float(self.beta2),
            "epsilon": float(self.epsilon),
        }

    def memories(self) -> dict[str, dict[tuple[int, str], np.ndarray]]:
        return {"means": self.means, "squares": self.squares}

    def state(self) -> dict[str, np.ndarray]:
        return {**super().state(), "steps": np.array(self.steps)}

    def load_state(self, state: dict[str, np.ndarray], layers: list[Dense]) -> None:
        state = dict(state)
        steps = state.pop("steps", None)
        if (
            not isinstance(steps, np.ndarray)
            or steps.shape != ()
            or steps.dtype.kind not in "iu"
            or steps < 0
        ):
            raise MyTorchError("optimizer state 'steps' must be a non-negative integer")
        means = {key.split("/", 1)[1] for key in state if key.startswith("means/")}
        squares = {key.split("/", 1)[1] for key in state if key.startswith("squares/")}
        if means != squares or (means and steps == 0):
            raise MyTorchError("optimizer state of Adam is inconsistent")
        super().load_state(state, layers)
        self.steps = int(steps)

    def compute_update(self, key: tuple[int, str], gradient: np.ndarray) -> np.ndarray:
        mean = self.beta1 * self.means.get(key, 0.0) + (1 - self.beta1) * gradient
        square = self.beta2 * self.squares.get(key, 0.0) + (1 - self.beta2) * gradient ** 2
        self.means[key] = mean
        self.squares[key] = square
        corrected_mean = mean / (1 - self.beta1 ** self.steps)
        corrected_square = square / (1 - self.beta2 ** self.steps)
        return self.learning_rate * corrected_mean / (np.sqrt(corrected_square) + self.epsilon)


OPTIMIZERS = {
    "sgd": SGD,
    "momentum": Momentum,
    "rmsprop": RMSProp,
    "adam": Adam,
}


def optimizer_name(optimizer: Optimizer) -> str:
    """Return the name under which an optimizer can be created again."""
    for name, optimizer_class in OPTIMIZERS.items():
        if type(optimizer) is optimizer_class:
            return name
    raise MyTorchError(f"unknown optimizer class '{type(optimizer).__name__}'")


def get_optimizer(name: str, **params: float | bool) -> Optimizer:
    """Return a new optimizer from its name and its parameters."""
    if name not in OPTIMIZERS:
        raise MyTorchError(
            f"unknown optimizer '{name}', expected one of: {', '.join(OPTIMIZERS)}"
        )
    try:
        return OPTIMIZERS[name](**params)
    except TypeError as error:
        raise MyTorchError(f"invalid parameters for optimizer '{name}': {error}") from error
