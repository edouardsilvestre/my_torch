"""Learning rate schedules: how the learning rate changes from one epoch to the next."""

import inspect
import math

import numpy as np

from my_torch.errors import MyTorchError

PLATEAU_MONITORED = ("loss", "accuracy", "val_loss", "val_accuracy")


def check_factor(value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value < 1:
        raise MyTorchError(f"factor must be between 0 and 1 excluded, got {value!r}")


def check_period(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise MyTorchError(f"{name} must be a positive integer, got {value!r}")


def check_minimum(value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value < np.inf:
        raise MyTorchError(f"minimum must be a non-negative number, got {value!r}")


class Schedule:
    """Give the learning rate of each epoch, from the learning rate of the optimizer.

    The learning rate of an epoch only depends on the epoch and on the history of the
    epochs before it: the same training always follows the same learning rates.
    """

    name = ""

    def learning_rate(
        self, base: float, epoch: int, epochs: int, history: dict[str, list[float]]
    ) -> float:
        """Return the learning rate of epoch (from 1) out of epochs, given the history of
        the previous epochs."""
        raise NotImplementedError

    def check_metrics(self, metrics: set[str]) -> None:
        """Check that the history will hold what the schedule needs."""

    def settings(self) -> dict[str, float | int | str]:
        """Return the name and the parameters of the schedule, to save them."""
        parameters = inspect.signature(type(self).__init__).parameters
        return {"name": self.name} | {
            name: getattr(self, name) for name in parameters if name != "self"
        }


class StepDecay(Schedule):
    """Multiply the learning rate by factor every `every` epochs."""

    name = "step"

    def __init__(self, factor: float = 0.5, every: int = 10) -> None:
        check_factor(factor)
        check_period("every", every)
        self.factor, self.every = float(factor), int(every)

    def learning_rate(self, base, epoch, epochs, history):
        return base * self.factor ** ((epoch - 1) // self.every)


class ExponentialDecay(Schedule):
    """Multiply the learning rate by factor at every epoch."""

    name = "exponential"

    def __init__(self, factor: float = 0.9) -> None:
        check_factor(factor)
        self.factor = float(factor)

    def learning_rate(self, base, epoch, epochs, history):
        return base * self.factor ** (epoch - 1)


class CosineAnnealing(Schedule):
    """Lower the learning rate from its value to minimum along half a cosine, over the
    epochs of the training."""

    name = "cosine"

    def __init__(self, minimum: float = 0.0) -> None:
        check_minimum(minimum)
        self.minimum = float(minimum)

    def learning_rate(self, base, epoch, epochs, history):
        if self.minimum > base:
            raise MyTorchError(
                f"the minimum learning rate {self.minimum} is above the learning rate {base}"
            )
        progress = (epoch - 1) / epochs
        return self.minimum + (base - self.minimum) * (1 + math.cos(math.pi * progress)) / 2


class ReduceOnPlateau(Schedule):
    """Multiply the learning rate by factor when the monitored metric has not improved for
    patience epochs, without going below minimum."""

    name = "plateau"

    def __init__(
        self, factor: float = 0.5, patience: int = 2, monitor: str = "val_loss",
        minimum: float = 0.0,
    ) -> None:
        check_factor(factor)
        check_period("patience", patience)
        if monitor not in PLATEAU_MONITORED:
            raise MyTorchError(
                f"monitor must be one of: {', '.join(PLATEAU_MONITORED)}, got {monitor!r}"
            )
        check_minimum(minimum)
        self.factor, self.patience = float(factor), int(patience)
        self.monitor, self.minimum = monitor, float(minimum)

    def check_metrics(self, metrics: set[str]) -> None:
        if self.monitor not in metrics:
            needs = ("a validation set" if self.monitor.startswith("val_") else "")
            if self.monitor.endswith("accuracy"):
                needs = (needs + " and " if needs else "") + "one-hot targets"
            raise MyTorchError(f"the plateau schedule on {self.monitor} needs {needs}")

    def learning_rate(self, base, epoch, epochs, history):
        # Replay the previous epochs: the learning rate is lowered each time the metric has
        # not improved for patience epochs in a row.
        rate, best, waiting = base, None, 0
        lower_is_better = self.monitor.endswith("loss")
        for value in history.get(self.monitor, [])[:epoch - 1]:
            if best is None or (value < best if lower_is_better else value > best):
                best, waiting = value, 0
                continue
            waiting += 1
            if waiting >= self.patience:
                rate, waiting = max(rate * self.factor, min(self.minimum, rate)), 0
        return rate


SCHEDULES: dict[str, type[Schedule]] = {
    schedule.name: schedule
    for schedule in (StepDecay, ExponentialDecay, CosineAnnealing, ReduceOnPlateau)
}


def get_schedule(name: str, **parameters) -> Schedule:
    """Create a schedule from its name and parameters."""
    if name not in SCHEDULES:
        raise MyTorchError(
            f"unknown learning rate schedule '{name}', expected one of: {', '.join(SCHEDULES)}"
        )
    accepted = schedule_parameters(name)
    unknown = sorted(set(parameters) - set(accepted))
    if unknown:
        raise MyTorchError(f"the {name} schedule has no parameter '{unknown[0]}'")
    return SCHEDULES[name](**parameters)


def schedule_from_settings(settings: dict | None) -> Schedule | None:
    """Create the schedule described by saved settings, {"name": ..., parameters}."""
    if settings is None:
        return None
    parameters = dict(settings)
    return get_schedule(parameters.pop("name"), **parameters)


def schedule_parameters(name: str) -> dict[str, type]:
    """Return the parameters of a schedule, with the type of their default."""
    signature = inspect.signature(SCHEDULES[name].__init__)
    return {
        parameter.name: type(parameter.default)
        for parameter in signature.parameters.values()
        if parameter.name != "self"
    }
