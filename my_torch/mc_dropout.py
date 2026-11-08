"""Monte Carlo dropout: several predictions with dropout active, averaged."""

import numpy as np

from my_torch.activations import Softmax
from my_torch.errors import MyTorchError
from my_torch.network import Network


def mc_dropout(
    network: Network,
    inputs: np.ndarray,
    passes: int,
    seed: int | np.random.Generator | None = None,
    softmax: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the network passes times with dropout active, and return the mean and the standard
    deviation of its outputs, for each example and each output.

    With softmax, the outputs are first turned into probabilities, as the scores of a
    network trained with cross_entropy should be. The mean is the prediction; the standard
    deviation measures how much the dropped neurons change it, an estimate of the
    uncertainty of the network on that example.
    """
    if isinstance(passes, bool) or not isinstance(passes, (int, np.integer)) or passes < 1:
        raise MyTorchError(f"passes must be a positive integer, got {passes!r}")
    if not any(network.dropout):
        raise MyTorchError("MC dropout needs a network with dropout: every pass would be equal")
    rng = np.random.default_rng(seed)
    total = squares = None
    for _ in range(passes):
        outputs = network.forward(np.asarray(inputs), rng)
        if softmax:
            outputs = Softmax().forward(outputs)
        total = outputs if total is None else total + outputs
        squares = outputs ** 2 if squares is None else squares + outputs ** 2
    mean = total / passes
    return mean, np.sqrt(np.maximum(squares / passes - mean ** 2, 0.0))
