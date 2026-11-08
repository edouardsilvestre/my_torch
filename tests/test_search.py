"""Tests of the hyperparameter search and of its use on the chess network."""

import json
import math

import numpy as np
import pytest

from my_torch import MyTorchError
from my_torch.search import (
    Choice, IntRange, LogUniform, Search, crossover, mutate, read_space, sample, significant,
)
from scripts.search_hyperparameters import configuration, main as search_main

SPACE = {"rate": LogUniform(1e-4, 1e-1), "size": IntRange(1, 9),
         "kind": Choice(["a", "b", "c"])}


def score(params):
    """Best at rate 1e-2, size 4 and kind 'a'."""
    return -(math.log10(params["rate"]) + 2) ** 2 - (params["size"] - 4) ** 2 / 4 + (
        params["kind"] == "a")


def test_samples_stay_in_their_space():
    rng = np.random.default_rng(0)
    for _ in range(200):
        params = sample(SPACE, rng)
        assert 1e-4 <= params["rate"] <= 1e-1 and params["rate"] == significant(params["rate"])
        assert params["size"] in range(1, 10) and isinstance(params["size"], int)
        assert params["kind"] in "abc"


def test_an_integer_range_includes_both_ends():
    rng = np.random.default_rng(0)
    assert {IntRange(1, 4).sample(rng) for _ in range(200)} == {1, 2, 3, 4}


def test_log_uniform_gives_each_power_of_ten_the_same_chances():
    rng = np.random.default_rng(0)
    powers = [math.floor(math.log10(LogUniform(1e-4, 1e-1).sample(rng))) for _ in range(3000)]
    counts = [powers.count(power) for power in (-4, -3, -2)]
    assert all(900 < count < 1100 for count in counts)


def test_significant_digits():
    assert significant(0.000123456) == 0.000123
    assert significant(123.456) == 123.0
    assert significant(0) == 0.0


def test_mutations_stay_in_their_space():
    rng = np.random.default_rng(0)
    for _ in range(200):
        assert Choice(["a", "b"]).mutate("a", rng) == "b"
        assert 1 <= IntRange(1, 3).mutate(1, rng) <= 3
        assert 1e-4 <= LogUniform(1e-4, 1e-3).mutate(1e-3, rng) <= 1e-3
    assert Choice(["a"]).mutate("a", rng) == "a"


def test_a_mutation_changes_at_least_one_hyperparameter():
    rng = np.random.default_rng(0)
    params = {"rate": 1e-2, "size": 4, "kind": "a"}
    for _ in range(100):
        assert mutate(params, SPACE, rng, rate=0.0) != params
    assert all(value != params[name]
               for name, value in mutate(params, SPACE, rng, rate=1.0).items() if name != "rate")


def test_a_crossover_takes_each_value_from_a_parent():
    rng = np.random.default_rng(0)
    first = {"rate": 1e-2, "size": 4, "kind": "a"}
    second = {"rate": 1e-3, "size": 7, "kind": "b"}
    children = [crossover(first, second, rng) for _ in range(100)]
    for child in children:
        assert all(child[name] in (first[name], second[name]) for name in first)
    assert any(child not in (first, second) for child in children)


@pytest.mark.parametrize("make, message", [
    (lambda: Choice([]), "non-empty list"),
    (lambda: LogUniform(0, 1), "0 < low < high"),
    (lambda: LogUniform(1, 0.1), "0 < low < high"),
    (lambda: IntRange(3, 3), "integers low < high"),
    (lambda: IntRange(1.5, 3), "integers low < high"),
    (lambda: Search({}, score), "at least one hyperparameter"),
    (lambda: Search({"rate": [1, 2]}, score), "needs a Choice, LogUniform or IntRange"),
])
def test_invalid_spaces_raise_an_error(make, message):
    with pytest.raises(MyTorchError, match=message):
        make()


def test_a_space_is_read_from_data():
    space = read_space({"rate": {"log_uniform": [1e-4, 1e-2]}, "size": {"int_range": [1, 4]},
                        "kind": {"choice": ["a", "b"]}})
    assert isinstance(space["rate"], LogUniform) and space["rate"].high == 1e-2
    assert isinstance(space["size"], IntRange) and space["size"].low == 1
    assert space["kind"].values == ["a", "b"]


@pytest.mark.parametrize("table, message", [
    (None, "at least one hyperparameter"),
    ({"rate": {"uniform": [0, 1]}}, "'rate' needs one of: choice, log_uniform, int_range"),
    ({"rate": {"log_uniform": [1e-4]}}, r"'rate': log_uniform needs \[low, high\]"),
    ({"rate": [1, 2]}, "'rate' needs one of"),
])
def test_invalid_space_data_raise_an_error(table, message):
    with pytest.raises(MyTorchError, match=message):
        read_space(table)


def counting(function):
    calls = []

    def evaluate(params):
        calls.append(params)
        return function(params)
    return evaluate, calls


def test_each_set_is_evaluated_once():
    evaluate, calls = counting(score)
    saved = []
    search = Search(SPACE, evaluate, on_trial=saved.append)
    params = {"rate": 1e-2, "size": 4, "kind": "a"}
    assert search.score(params, "manual") == search.score(dict(params), "manual") == 1.0
    assert len(calls) == len(search.trials) == len(saved) == 1
    assert saved[0].params == params and saved[0].number == 1


def test_the_ranking_puts_the_best_first_and_the_first_on_a_tie():
    search = Search(SPACE, lambda params: params["size"] // 2)
    for size in (3, 5, 4, 2):
        search.score({"rate": 1e-2, "size": size, "kind": "a"}, "manual")
    assert [trial.params["size"] for trial in search.ranking()] == [5, 4, 3, 2]


def test_details_are_kept_with_the_trial():
    search = Search(SPACE, lambda params: (0.5, {"kept_epoch": 3}))
    search.score({"rate": 1e-2, "size": 4, "kind": "a"}, "manual")
    assert (search.trials[0].score, search.trials[0].details) == (0.5, {"kept_epoch": 3})


def test_a_score_must_be_a_number():
    with pytest.raises(MyTorchError, match="must be a number"):
        Search(SPACE, lambda params: float("nan")).random(1, seed=0)


def test_random_search_is_reproducible():
    first, second = Search(SPACE, score), Search(SPACE, score)
    first.random(10, seed=3)
    second.random(10, seed=3)
    assert [trial.params for trial in first.trials] == [trial.params for trial in second.trials]
    assert len(first.trials) == 10
    assert all(trial.method == "random" for trial in first.trials)


def test_a_restored_search_is_not_evaluated_again():
    first = Search(SPACE, score)
    first.random(6, seed=1)
    evaluate, calls = counting(score)
    second = Search(SPACE, evaluate)
    second.restore(first.trials[:4])
    second.random(6, seed=1)
    assert len(calls) == 2
    assert [trial.params for trial in second.trials] == [trial.params for trial in first.trials]
    assert [trial.number for trial in second.trials] == [1, 2, 3, 4, 5, 6]


def test_the_genetic_search_keeps_its_elite_and_improves():
    search = Search(SPACE, score)
    trials = search.genetic(population=10, generations=6, seed=0)
    best = [max(trial.score for trial in trials if trial.generation <= generation)
            for generation in range(1, 7)]
    first = max(trial.score for trial in trials if trial.generation == 1)
    assert best[-1] > first
    # With elite sets kept, the best of each generation is never lost.
    assert search.ranking()[0].score == best[-1] > 0.5


@pytest.mark.parametrize("seed", range(5))
def test_the_elite_goes_to_the_next_generation(seed):
    search = Search(SPACE, score)
    search.genetic(population=6, generations=5, seed=seed, elite=1)
    bests = [max(scores) for scores in search.generations]
    assert len(bests) == 5
    assert all(later >= earlier for earlier, later in zip(bests, bests[1:]))


def test_the_genetic_search_is_reproducible():
    first, second = Search(SPACE, score), Search(SPACE, score)
    first.genetic(6, 3, seed=2)
    second.genetic(6, 3, seed=2)
    assert [trial.params for trial in first.trials] == [trial.params for trial in second.trials]
    assert {trial.generation for trial in first.trials} <= {1, 2, 3}


@pytest.mark.parametrize("options, message", [
    ({"population": 0}, "population must be a positive integer"),
    ({"generations": 0}, "generations must be a positive integer"),
    ({"elite": 4}, "elite must be from 0 to population - 1"),
    ({"mutation_rate": 1.5}, r"mutation_rate must be in \[0, 1\]"),
    ({"tournament": 0}, "tournament must be a positive integer"),
])
def test_invalid_genetic_settings_raise_an_error(options, message):
    settings = {"population": 4, "generations": 2, **options}
    with pytest.raises(MyTorchError, match=message):
        Search(SPACE, score).genetic(**settings)


BASE = {
    "network": {
        "input_size": 769, "seed": 0,
        "layers": [{"size": 32, "activation": "relu"}, {"size": 5, "activation": "identity"}],
    },
    "training": {"loss": "cross_entropy", "epochs": 24, "batch_size": 64, "seed": 0},
    "optimizer": {"name": "adam", "learning_rate": 0.001},
}


def test_the_searched_values_replace_those_of_the_base():
    params = {"hidden": [16, 8], "activation": "tanh", "dropout": 0.1, "learning_rate": 0.01,
              "batch_size": 32, "l2": 0.0001}
    data = configuration(BASE, params, epochs=5, patience=2)
    assert data["network"]["layers"] == [
        {"size": 16, "activation": "tanh", "dropout": 0.1},
        {"size": 8, "activation": "tanh", "dropout": 0.1},
        {"size": 5, "activation": "identity"},
    ]
    assert (data["training"]["batch_size"], data["training"]["l2"]) == (32, 0.0001)
    assert (data["training"]["epochs"], data["training"]["patience"]) == (5, 2)
    assert data["training"]["monitor"] == "val_accuracy"
    assert data["optimizer"]["learning_rate"] == 0.01
    assert BASE["training"]["epochs"] == 24


def test_values_not_searched_come_from_the_base():
    data = configuration(BASE, {"learning_rate": 0.01}, epochs=5, patience=2)
    assert data["network"]["layers"][0] == {"size": 32, "activation": "relu", "dropout": 0.0}
    assert data["training"]["batch_size"] == 64


def test_an_unknown_hyperparameter_is_refused():
    with pytest.raises(MyTorchError, match="unknown hyperparameter 'momentum'"):
        configuration(BASE, {"momentum": 0.9}, epochs=5, patience=2)


FENS = [
    "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3 Checkmate Black",
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 Nothing",
    "rnbqkbnr/pppp2pp/8/4pp1Q/3P4/4P3/PPP2PPP/RNB1KBNR b KQkq - 1 3 Check White",
    "8/8/8/8/8/8/8/k1K5 w - - 0 1 Nothing",
]


@pytest.fixture
def files(tmp_path):
    (tmp_path / "boards.txt").write_text("\n".join(FENS) + "\n")
    (tmp_path / "base.conf").write_text("""[network]
input_size = 769
layers = [{ size = 8, activation = "relu" }, { size = 5, activation = "identity" }]
seed = 0

[training]
loss = "cross_entropy"
epochs = 50
batch_size = 4
seed = 0

[optimizer]
name = "adam"
""")
    (tmp_path / "space.toml").write_text(
        '[space]\nlearning_rate = { log_uniform = [0.001, 0.1] }\n'
        'hidden = { choice = [[8], [16]] }\n')
    return tmp_path


def run(files, *method):
    search_main(["--space", str(files / "space.toml"), "--base", str(files / "base.conf"),
                 "--train", str(files / "boards.txt"), "--validation", str(files / "boards.txt"),
                 "--output", str(files / "trials.jsonl"), "--epochs", "4", *method])


def test_a_search_saves_and_ranks_its_trials(files, capsys):
    run(files, "random", "--trials", "3")
    trials = [json.loads(line) for line in (files / "trials.jsonl").read_text().splitlines()]
    assert [trial["number"] for trial in trials] == [1, 2, 3]
    assert all(trial["details"]["epochs_run"] <= 4 for trial in trials)
    output = capsys.readouterr().out
    assert output.count("\ntrial ") + output.startswith("trial ") == 3
    assert "| Rank | Trial | Method | learning_rate | hidden | Val. acc." in output
    ranked = sorted(trials, key=lambda trial: (-trial["score"], trial["number"]))
    assert f"| 1 | {ranked[0]['number']} | random |" in output


def test_a_search_goes_on_where_it_stopped(files, capsys):
    run(files, "random", "--trials", "2")
    first = (files / "trials.jsonl").read_text()
    run(files, "random", "--trials", "3")
    lines = (files / "trials.jsonl").read_text().splitlines()
    assert (files / "trials.jsonl").read_text().startswith(first)
    assert len(lines) == 3


def test_a_genetic_search_on_the_chess_network(files, capsys):
    run(files, "genetic", "--population", "3", "--generations", "2", "--elite", "1")
    trials = [json.loads(line) for line in (files / "trials.jsonl").read_text().splitlines()]
    assert {trial["method"] for trial in trials} == {"genetic"}
    assert {trial["generation"] for trial in trials} <= {1, 2}


def test_a_trial_is_scored_at_its_kept_epoch(files, monkeypatch, capsys):
    import scripts.search_hyperparameters as searching
    history = {"loss": [0.9, 0.5, 0.6, 0.7], "accuracy": [0.5, 0.8, 0.9, 0.95],
               "val_loss": [0.9, 0.6, 0.7, 0.8], "val_accuracy": [0.3, 0.6, 0.5, 0.55]}
    monkeypatch.setattr(searching, "train", lambda *arguments, **options: history)
    run(files, "random", "--trials", "1")
    trial = json.loads((files / "trials.jsonl").read_text())
    assert trial["score"] == 0.6
    assert (trial["details"]["kept_epoch"], trial["details"]["train_accuracy"]) == (2, 0.8)


def test_an_invalid_space_stops_with_a_message(files):
    (files / "space.toml").write_text('[space]\nmomentum = { choice = [0.9] }\n')
    with pytest.raises(SystemExit) as error:
        run(files, "random", "--trials", "1")
    assert "unknown hyperparameter 'momentum'" in str(error.value.code)
