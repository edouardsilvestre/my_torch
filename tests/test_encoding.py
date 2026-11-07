"""Tests of the encoding of chess positions as input vectors."""

import numpy as np
import pytest

from analyzer.encoding import (
    ACTIVE_COLOR_INPUT, INPUT_SIZE, encode_position, encode_positions, input_index,
)
from analyzer.fen import PIECES, parse_fen
from my_torch import Network, get_loss, get_optimizer, load_config, train

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def encode(fen):
    return encode_position(parse_fen(fen))


def decode(vector):
    """Rebuild the board and the side to move from a vector, to check the encoding."""
    planes = vector[:768].reshape(12, 64)
    board = []
    for row in range(8):
        rank = []
        for file in range(8):
            pieces = np.flatnonzero(planes[:, row * 8 + file])
            assert len(pieces) <= 1
            rank.append(PIECES[pieces[0]] if len(pieces) else None)
        board.append(tuple(rank))
    return tuple(board), "w" if vector[ACTIVE_COLOR_INPUT] == 1 else "b"


def random_fen(rng):
    ranks = []
    for _ in range(8):
        text, empty = "", 0
        for _ in range(8):
            if rng.random() < 0.3:
                text += (str(empty) if empty else "") + PIECES[rng.integers(12)]
                empty = 0
            else:
                empty += 1
        ranks.append(text + (str(empty) if empty else ""))
    return "/".join(ranks) + (" w" if rng.random() < 0.5 else " b") + " - - 0 1"


def test_sizes():
    assert INPUT_SIZE == 769
    assert ACTIVE_COLOR_INPUT == 768


def test_vector_is_made_of_bytes():
    vector = encode(START)
    assert vector.shape == (769,)
    assert vector.dtype == np.uint8
    assert set(np.unique(vector)) <= {0, 1}


def test_initial_position_by_hand():
    vector = encode(START)
    assert vector.sum() == 33
    assert list(np.flatnonzero(vector[0:64])) == list(range(48, 56))
    assert list(np.flatnonzero(vector[6 * 64:7 * 64])) == list(range(8, 16))
    assert vector[5 * 64 + 60] == 1
    assert vector[11 * 64 + 4] == 1
    assert vector[3 * 64 + 56] == 1 and vector[3 * 64 + 63] == 1
    assert vector[768] == 1


def test_almost_empty_board_by_hand():
    vector = encode("8/8/8/8/8/8/8/k1K5 b - - 0 1")
    assert list(np.flatnonzero(vector)) == [5 * 64 + 58, 11 * 64 + 56]


@pytest.mark.parametrize("color, expected", [("w", 1), ("b", 0)])
def test_side_to_move(color, expected):
    assert encode(f"8/8/8/8/8/8/8/8 {color} - - 0 1")[ACTIVE_COLOR_INPUT] == expected


@pytest.mark.parametrize("piece, row, file, expected", [
    ("P", 0, 0, 0),
    ("P", 7, 7, 63),
    ("N", 0, 0, 64),
    ("K", 7, 4, 5 * 64 + 60),
    ("p", 0, 0, 6 * 64),
    ("k", 0, 4, 11 * 64 + 4),
    ("k", 7, 7, 767),
])
def test_input_index(piece, row, file, expected):
    assert input_index(piece, row, file) == expected


def test_each_piece_and_square_has_its_own_input():
    indices = {input_index(piece, row, file)
               for piece in PIECES for row in range(8) for file in range(8)}
    assert indices == set(range(768))


@pytest.mark.parametrize("square, row, file", [("a8", 0, 0), ("h8", 0, 7), ("a1", 7, 0),
                                               ("e4", 4, 4), ("h1", 7, 7)])
def test_squares_follow_the_reading_order_of_the_fen(square, row, file):
    rank = 8 - int(square[1])
    placement = ["8"] * 8
    column = ord(square[0]) - ord("a")
    before = str(column) if column else ""
    after = str(7 - column) if column < 7 else ""
    placement[rank] = before + "Q" + after
    vector = encode("/".join(placement) + " w - - 0 1")
    assert list(np.flatnonzero(vector[:768])) == [input_index("Q", row, file)]


def test_ignored_fields_do_not_change_the_vector():
    reference = encode(START)
    for fen in ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1",
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq e3 0 1",
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 42 99"):
        assert np.array_equal(encode(fen), reference)


def test_decoding_gives_back_random_boards():
    rng = np.random.default_rng(0)
    for _ in range(300):
        position = parse_fen(random_fen(rng))
        assert decode(encode_position(position)) == (position.board, position.active_color)


def test_batch_encoding_stacks_the_vectors():
    rng = np.random.default_rng(1)
    positions = [parse_fen(random_fen(rng)) for _ in range(50)]
    batch = encode_positions(positions)
    assert batch.shape == (50, 769) and batch.dtype == np.uint8
    for row, position in enumerate(positions):
        assert np.array_equal(batch[row], encode_position(position))


def test_batch_encoding_of_nothing():
    assert encode_positions([]).shape == (0, 769)


def test_bytes_give_the_same_training_as_decimals():
    rng = np.random.default_rng(2)
    inputs = encode_positions(parse_fen(random_fen(rng)) for _ in range(200))
    targets = np.eye(3)[rng.integers(0, 3, 200)]
    results = []
    for batch in (inputs, inputs.astype(np.float64)):
        network = Network(INPUT_SIZE, [(16, "relu"), (3, "identity")], seed=0)
        history = train(network, get_loss("cross_entropy"), get_optimizer("adam"),
                        batch, targets, 2, 32, seed=0)
        results.append((history, network))
    assert results[0][0] == results[1][0]
    for first, second in zip(results[0][1].layers, results[1][1].layers):
        assert np.array_equal(first.weights, second.weights)


def test_example_configuration_matches_the_encoding():
    config = load_config("configs/basic_network.conf")
    assert config.input_size == INPUT_SIZE
    config.create_network().forward(encode_positions([parse_fen(START)]))
