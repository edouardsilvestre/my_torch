"""Measure how the learning rate affects the optimizers on two toy problems.

Run from the root of the repository: python3 -m scripts.optimizer_learning_rates
"""

import numpy as np

from my_torch import Network, get_loss, get_optimizer

XOR_INPUTS = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
XOR_LABELS = np.array([0, 1, 1, 0])
XOR_TARGETS = np.eye(2)[XOR_LABELS]
XOR_ARCHITECTURE = [(4, "tanh"), (2, "identity")]
XOR_EPOCHS = 2000

REGRESSION_INPUTS = np.array(
    [[a, b] for a in np.linspace(-1.0, 1.0, 5) for b in np.linspace(-1.0, 1.0, 5)]
)
REGRESSION_WEIGHTS = np.array([[2.0], [-3.0]])
REGRESSION_BIAS = 1.0
REGRESSION_TARGETS = REGRESSION_INPUTS @ REGRESSION_WEIGHTS + REGRESSION_BIAS
REGRESSION_EPOCHS = 2000

DIVERGENCE_EPOCHS = 500
DIVERGENCE_LOSS = 1e6
CONVERGENCE_LOSS = 1e-6

SWEEP_OPTIMIZERS = [
    ("SGD", "sgd", {}),
    ("Momentum", "momentum", {}),
    ("Nesterov", "momentum", {"nesterov": True}),
]
REGRESSION_RATES = [0.5, 0.6, 0.7, 0.9, 1.1, 1.5, 1.8, 2.0]
XOR_RATES = [0.5, 1, 2, 5, 10]
XOR_SWEEP_SEEDS = 30

TEST_SETTINGS = [
    ("SGD", "sgd", {}, 0.5, 0.1),
    ("Momentum", "momentum", {}, 0.05, 0.05),
    ("Nesterov", "momentum", {"nesterov": True}, 0.05, 0.05),
    ("RMSProp", "rmsprop", {}, 0.1, 0.01),
    ("Adam", "adam", {}, 0.01, 0.01),
]
XOR_SETTINGS_SEEDS = 100
REGRESSION_SETTINGS_SEEDS = 20


def train(network, optimizer, loss, inputs, targets, epochs):
    """Train a network on a full batch, then return the final loss."""
    for _ in range(epochs):
        network.backward(loss.backward(network.forward(inputs), targets))
        optimizer.step(network.layers)
    return loss.forward(network.forward(inputs), targets)


def learns_xor(name, params, learning_rate, seed):
    """Return whether a network gives the 4 right answers of XOR after training."""
    network = Network(2, XOR_ARCHITECTURE, "xavier_normal", seed=seed)
    optimizer = get_optimizer(name, learning_rate=learning_rate, **params)
    train(network, optimizer, get_loss("cross_entropy"), XOR_INPUTS, XOR_TARGETS, XOR_EPOCHS)
    return np.array_equal(network.predict(XOR_INPUTS), XOR_LABELS)


def regression_outcome(name, params, learning_rate):
    """Describe how a regression ends with a given learning rate."""
    network = Network(2, [(1, "identity")], seed=0)
    optimizer = get_optimizer(name, learning_rate=learning_rate, **params)
    loss = train(
        network, optimizer, get_loss("mse"),
        REGRESSION_INPUTS, REGRESSION_TARGETS, DIVERGENCE_EPOCHS,
    )
    if not np.isfinite(loss) or loss > DIVERGENCE_LOSS:
        return "Diverge"
    return "Converge" if loss < CONVERGENCE_LOSS else "Indécis"


def regression_error(name, params, learning_rate, seed):
    """Return the largest distance between the learned parameters and the solution."""
    network = Network(2, [(1, "identity")], seed=seed)
    optimizer = get_optimizer(name, learning_rate=learning_rate, **params)
    train(
        network, optimizer, get_loss("mse"),
        REGRESSION_INPUTS, REGRESSION_TARGETS, REGRESSION_EPOCHS,
    )
    layer = network.layers[0]
    return max(
        np.abs(layer.weights - REGRESSION_WEIGHTS).max(),
        np.abs(layer.biases - REGRESSION_BIAS).max(),
    )


def print_header(first_column):
    names = " | ".join(name for name, _, _ in SWEEP_OPTIMIZERS)
    print(f"| {first_column} | {names} |")
    print("|---|" + "---|" * len(SWEEP_OPTIMIZERS))


def print_regression_sweep():
    print(f"Régression, {DIVERGENCE_EPOCHS} pas, graine 0 :\n")
    print_header("Learning rate")
    for rate in REGRESSION_RATES:
        outcomes = [regression_outcome(name, params, rate) for _, name, params in SWEEP_OPTIMIZERS]
        print(f"| {rate} | " + " | ".join(outcomes) + " |")


def print_xor_sweep():
    print(f"\nXOR, {XOR_EPOCHS} epochs, graines réussies sur {XOR_SWEEP_SEEDS} :\n")
    print_header("Learning rate")
    for rate in XOR_RATES:
        counts = [
            sum(learns_xor(name, params, rate, seed) for seed in range(XOR_SWEEP_SEEDS))
            for _, name, params in SWEEP_OPTIMIZERS
        ]
        print(f"| {rate} | " + " | ".join(map(str, counts)) + " |")


def print_test_settings():
    print("\nRéglages utilisés par les tests :\n")
    print(f"| Optimiseur | XOR : learning rate | Réussite sur {XOR_SETTINGS_SEEDS} graines "
          f"| Régression : learning rate | Pire écart sur {REGRESSION_SETTINGS_SEEDS} graines |")
    print("|---|---|---|---|---|")
    for label, name, params, xor_rate, regression_rate in TEST_SETTINGS:
        successes = sum(
            learns_xor(name, params, xor_rate, seed) for seed in range(XOR_SETTINGS_SEEDS)
        )
        worst = max(
            regression_error(name, params, regression_rate, seed)
            for seed in range(REGRESSION_SETTINGS_SEEDS)
        )
        print(f"| {label} | {xor_rate} | {successes} | {regression_rate} | {worst:.1e} |")


def main() -> None:
    with np.errstate(all="ignore"):
        print_regression_sweep()
        print_xor_sweep()
        print_test_settings()


if __name__ == "__main__":
    main()
