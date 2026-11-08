"""Time the matrix product of a dense layer: NumPy against a naive Python implementation.

Run from the root of the repository:  python3 -m scripts.numpy_speed
The product is the forward of the first layer of the chess network, a batch of 64 positions
of 769 inputs times 256 neurons, the most costly product of a training step.
"""

import argparse
import sys
import time

import numpy as np


def naive_matmul(left: list[list[float]], right: list[list[float]]) -> list[list[float]]:
    """The textbook triple loop, in pure Python."""
    columns = list(zip(*right))
    return [[sum(a * b for a, b in zip(row, column)) for column in columns] for row in left]


def best_time(function, repeats: int) -> float:
    """The shortest of several runs, the least disturbed by the rest of the machine."""
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        times.append(time.perf_counter() - start)
    return min(times)


def measure(batch: int, inputs: int, neurons: int, repeats: int) -> dict[str, float]:
    rng = np.random.default_rng(0)
    left, right = rng.random((batch, inputs)), rng.random((inputs, neurons))
    left_lists, right_lists = left.tolist(), right.tolist()
    if not np.allclose(naive_matmul(left_lists, right_lists), left @ right):
        raise AssertionError("the naive product differs from NumPy")
    naive = best_time(lambda: naive_matmul(left_lists, right_lists), repeats)
    vectorised = best_time(lambda: left @ right, max(repeats, 20))
    return {"naive": naive, "numpy": vectorised, "speedup": naive / vectorised,
            "operations": 2 * batch * inputs * neurons}


def parse_arguments(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--inputs", type=int, default=769)
    parser.add_argument("--neurons", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=3, help="runs of the naive product")
    return parser.parse_args(argv)


def main(argv: list[str]) -> None:
    arguments = parse_arguments(argv)
    result = measure(arguments.batch, arguments.inputs, arguments.neurons, arguments.repeats)
    print(f"({arguments.batch} x {arguments.inputs}) @ ({arguments.inputs} x "
          f"{arguments.neurons}): {result['operations']:,} operations")
    print(f"naive Python: {result['naive'] * 1000:10.3f} ms "
          f"({result['operations'] / result['naive'] / 1e9:.3f} Gflop/s)")
    print(f"NumPy:        {result['numpy'] * 1000:10.3f} ms "
          f"({result['operations'] / result['numpy'] / 1e9:.3f} Gflop/s)")
    print(f"NumPy is {result['speedup']:,.0f} times faster")


if __name__ == "__main__":
    main(sys.argv[1:])
