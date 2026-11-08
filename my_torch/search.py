"""Hyperparameter search: a random search and a genetic algorithm over a search space.

A search space maps the name of each hyperparameter to the values it can take: a Choice
among listed values, a LogUniform range for rates, or an IntRange. The search calls an
evaluate function on sets of hyperparameters, and keeps the best score: higher is better,
like a validation accuracy. Each set is evaluated once, even if the search meets it again.
"""

import json
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from my_torch.errors import MyTorchError


def significant(value: float, digits: int = 3) -> float:
    """Round to a few significant digits, so that a sampled rate stays readable."""
    if value == 0:
        return 0.0
    return round(value, digits - 1 - math.floor(math.log10(abs(value))))


class Choice:
    """One of a list of values, which can be of any JSON type."""

    def __init__(self, values: list) -> None:
        if not isinstance(values, list) or not values:
            raise MyTorchError(f"a choice needs a non-empty list of values, got {values!r}")
        self.values = values

    def sample(self, rng: np.random.Generator) -> Any:
        return self.values[rng.integers(len(self.values))]

    def mutate(self, value: Any, rng: np.random.Generator) -> Any:
        """Another value of the list, when there is one."""
        others = [other for other in self.values if other != value]
        return others[rng.integers(len(others))] if others else value


class LogUniform:
    """A positive number whose logarithm is uniform between those of low and high: each
    power of ten gets the same chances, as fits a learning rate."""

    def __init__(self, low: float, high: float) -> None:
        if not 0 < low < high < math.inf:
            raise MyTorchError(f"a log-uniform range needs 0 < low < high, got {low}, {high}")
        self.low, self.high = float(low), float(high)

    def sample(self, rng: np.random.Generator) -> float:
        return significant(math.exp(rng.uniform(math.log(self.low), math.log(self.high))))

    def mutate(self, value: float, rng: np.random.Generator) -> float:
        """Multiply by a random factor, about 0.5 to 2, within the range."""
        moved = value * math.exp(rng.normal(0.0, 0.5))
        return significant(min(max(moved, self.low), self.high))


class IntRange:
    """An integer from low to high, both included."""

    def __init__(self, low: int, high: int) -> None:
        if not isinstance(low, int) or not isinstance(high, int) or not low < high:
            raise MyTorchError(f"an integer range needs integers low < high, got {low}, {high}")
        self.low, self.high = low, high

    def sample(self, rng: np.random.Generator) -> int:
        return int(rng.integers(self.low, self.high + 1))

    def mutate(self, value: int, rng: np.random.Generator) -> int:
        """A neighbouring value, one or two steps away, within the range."""
        step = int(rng.choice([-2, -1, 1, 2]))
        return min(max(value + step, self.low), self.high)


Dimension = Choice | LogUniform | IntRange


DIMENSIONS = {"choice": Choice, "log_uniform": LogUniform, "int_range": IntRange}


def read_space(table: Any) -> dict[str, Dimension]:
    """Read a search space written as data, for example in TOML:
    { learning_rate = { log_uniform = [0.0001, 0.01] }, batch_size = { choice = [32, 64] } }"""
    if not isinstance(table, dict) or not table:
        raise MyTorchError("a search space needs at least one hyperparameter")
    space = {}
    for name, description in table.items():
        if not isinstance(description, dict) or len(description) != 1 or (
                next(iter(description)) not in DIMENSIONS):
            raise MyTorchError(f"hyperparameter '{name}' needs one of: {', '.join(DIMENSIONS)}")
        kind, values = next(iter(description.items()))
        if kind == "choice":
            space[name] = Choice(values)
        elif not isinstance(values, list) or len(values) != 2:
            raise MyTorchError(f"hyperparameter '{name}': {kind} needs [low, high]")
        else:
            space[name] = DIMENSIONS[kind](*values)
    return space


def check_space(space: dict[str, Dimension]) -> None:
    if not isinstance(space, dict) or not space:
        raise MyTorchError("a search space needs at least one hyperparameter")
    for name, dimension in space.items():
        if not isinstance(dimension, (Choice, LogUniform, IntRange)):
            raise MyTorchError(f"hyperparameter '{name}' needs a Choice, LogUniform or IntRange")


def sample(space: dict[str, Dimension], rng: np.random.Generator) -> dict[str, Any]:
    """Draw every hyperparameter at random."""
    return {name: dimension.sample(rng) for name, dimension in space.items()}


def crossover(first: dict, second: dict, rng: np.random.Generator) -> dict[str, Any]:
    """Take each hyperparameter from one parent or the other, with equal chances."""
    return {name: (first if rng.random() < 0.5 else second)[name] for name in first}


def mutate(params: dict, space: dict[str, Dimension], rng: np.random.Generator,
           rate: float) -> dict[str, Any]:
    """Change each hyperparameter with probability rate, and at least one of them."""
    names = list(space)
    changed = [name for name in names if rng.random() < rate]
    if not changed:
        changed = [names[rng.integers(len(names))]]
    return {name: space[name].mutate(value, rng) if name in changed else value
            for name, value in params.items()}


@dataclass
class Trial:
    """One evaluated set of hyperparameters."""

    number: int
    params: dict[str, Any]
    score: float
    method: str
    generation: int | None = None
    details: dict[str, Any] = field(default_factory=dict)


class Search:
    """Evaluate sets of hyperparameters once each, and record every trial.

    evaluate returns the score of a set, or a (score, details) pair where details are kept
    with the trial, like the epoch kept or the time taken. on_trial is called after each new
    trial, for example to save it at once: a long search interrupted keeps its trials.
    """

    def __init__(self, space: dict[str, Dimension],
                 evaluate: Callable[[dict[str, Any]], float | tuple[float, dict]],
                 on_trial: Callable[[Trial], None] | None = None) -> None:
        check_space(space)
        self.space, self.evaluate, self.on_trial = space, evaluate, on_trial
        self.trials: list[Trial] = []
        self.scores: dict[str, float] = {}
        # Scores of the members of each generation of the genetic search, in order.
        self.generations: list[list[float]] = []

    def score(self, params: dict[str, Any], method: str, generation: int | None = None) -> float:
        key = json.dumps(params, sort_keys=True)
        if key not in self.scores:
            result = self.evaluate(dict(params))
            score, details = result if isinstance(result, tuple) else (result, {})
            if not isinstance(score, (int, float)) or math.isnan(score):
                raise MyTorchError(f"the score of a trial must be a number, got {score!r}")
            self.scores[key] = float(score)
            trial = Trial(len(self.trials) + 1, dict(params), float(score), method, generation,
                          details)
            self.trials.append(trial)
            if self.on_trial is not None:
                self.on_trial(trial)
        return self.scores[key]

    def ranking(self) -> list[Trial]:
        """The trials from the best score to the worst; the first trial wins a tie."""
        return sorted(self.trials, key=lambda trial: (-trial.score, trial.number))

    def restore(self, trials: list[Trial]) -> None:
        """Take back the trials of an earlier run of the same search, which are then not
        evaluated again: an interrupted search goes on where it stopped."""
        for trial in trials:
            self.scores[json.dumps(trial.params, sort_keys=True)] = trial.score
            self.trials.append(trial)

    def random(self, trials: int, seed: int | None = None) -> list[Trial]:
        """Draw trials sets at random and evaluate them. A set drawn twice is evaluated
        once: the search can give fewer trials than draws."""
        check_positive("trials", trials)
        rng = np.random.default_rng(seed)
        start = len(self.trials)
        for _ in range(trials):
            self.score(sample(self.space, rng), "random")
        return self.trials[start:]

    def genetic(self, population: int, generations: int, seed: int | None = None,
                mutation_rate: float = 0.3, elite: int = 2, tournament: int = 3) -> list[Trial]:
        """Evolve a population of sets of hyperparameters.

        The first generation is drawn at random. Each next one keeps the elite best sets,
        and fills the rest with children: two parents, each the best of tournament sets
        drawn from the generation, give a child by crossover, then mutation.
        """
        check_positive("population", population)
        check_positive("generations", generations)
        check_positive("tournament", tournament)
        if not 0 <= elite < population:
            raise MyTorchError(f"elite must be from 0 to population - 1, got {elite!r}")
        if not 0 <= mutation_rate <= 1:
            raise MyTorchError(f"mutation_rate must be in [0, 1], got {mutation_rate!r}")
        rng = np.random.default_rng(seed)
        start = len(self.trials)
        members = [sample(self.space, rng) for _ in range(population)]
        for generation in range(1, generations + 1):
            scores = [self.score(member, "genetic", generation) for member in members]
            self.generations.append(scores)
            if generation == generations:
                break
            order = sorted(range(population), key=lambda index: -scores[index])
            children = [members[index] for index in order[:elite]]
            while len(children) < population:
                first, second = (members[self.pick(scores, tournament, rng)] for _ in range(2))
                children.append(mutate(crossover(first, second, rng), self.space, rng,
                                       mutation_rate))
            members = children
        return self.trials[start:]

    @staticmethod
    def pick(scores: list[float], tournament: int, rng: np.random.Generator) -> int:
        """The index of the best of tournament members drawn at random."""
        drawn = rng.integers(len(scores), size=tournament)
        return int(max(drawn, key=lambda index: scores[index]))


def check_positive(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise MyTorchError(f"{name} must be a positive integer, got {value!r}")
