"""Measure how often small networks learn XOR, over many random seeds.

Run from the root of the repository: python3 -m scripts.xor_convergence
"""

import numpy as np

from my_torch import Network, get_loss, get_optimizer

INPUTS = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
LABELS = np.array([0, 1, 1, 0])
TARGETS = np.eye(2)[LABELS]
SEEDS = 100
LEARNING_RATE = 0.5
EPOCHS = 2000
MAX_LISTED_FAILURES = 10
CONFIGURATIONS = [
    ("Une seule couche linéaire", [(2, "identity")], "he"),
    ("2 neurones `relu`", [(2, "relu"), (2, "identity")], "he"),
    ("2 neurones `tanh`", [(2, "tanh"), (2, "identity")], "xavier_normal"),
    ("4 neurones `relu`", [(4, "relu"), (2, "identity")], "he"),
    ("8 neurones `relu`", [(8, "relu"), (2, "identity")], "he"),
    ("4 neurones `tanh`", [(4, "tanh"), (2, "identity")], "xavier_normal"),
]


def train(architecture: list[tuple[int, str]], initializer: str, seed: int) -> tuple[bool, float]:
    """Train a network with a plain gradient descent, return its success and final loss."""
    loss = get_loss("cross_entropy")
    optimizer = get_optimizer("sgd", learning_rate=LEARNING_RATE)
    network = Network(2, architecture, initializer, seed=seed)
    for _ in range(EPOCHS):
        network.backward(loss.backward(network.forward(INPUTS), TARGETS))
        optimizer.step(network.layers)
    success = np.array_equal(network.predict(INPUTS), LABELS)
    return success, loss.forward(network.forward(INPUTS), TARGETS)


def main() -> None:
    print("| Architecture | Réussite | Coût médian | Graines en échec |")
    print("|---|---|---|---|")
    for name, architecture, initializer in CONFIGURATIONS:
        results = [train(architecture, initializer, seed) for seed in range(SEEDS)]
        failures = [seed for seed, (success, _) in enumerate(results) if not success]
        median = np.median([loss for _, loss in results])
        listed = ", ".join(map(str, failures)) if len(failures) <= MAX_LISTED_FAILURES else "..."
        print(f"| {name} | {SEEDS - len(failures)} / {SEEDS} | {median:.4f} | {listed} |")


if __name__ == "__main__":
    main()
