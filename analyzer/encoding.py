"""Encoding of chess positions as input vectors for the network."""

from collections.abc import Iterable

import numpy as np

from analyzer.fen import PIECES, Position

SQUARES = 64
PIECE_INPUTS = len(PIECES) * SQUARES
ACTIVE_COLOR_INPUT = PIECE_INPUTS
INPUT_SIZE = PIECE_INPUTS + 1
PIECE_PLANES = {piece: index for index, piece in enumerate(PIECES)}


def input_index(piece: str, row: int, file: int) -> int:
    """Index of the input telling whether a piece stands on a square.

    Rows are counted in the reading order of the FEN: row 0 is rank 8, and file 0 is
    file a. Square a8 is therefore the first of each plane, and h1 the last.
    """
    return PIECE_PLANES[piece] * SQUARES + row * 8 + file


def encode_into(position: Position, vector: np.ndarray) -> None:
    for row, rank in enumerate(position.board):
        for file, piece in enumerate(rank):
            if piece is not None:
                vector[input_index(piece, row, file)] = 1
    vector[ACTIVE_COLOR_INPUT] = 1 if position.active_color == "w" else 0


def encode_position(position: Position) -> np.ndarray:
    """Return the 769 inputs of a position: 12 planes of 64 squares, then the side to move."""
    vector = np.zeros(INPUT_SIZE, dtype=np.uint8)
    encode_into(position, vector)
    return vector


def encode_positions(positions: Iterable[Position]) -> np.ndarray:
    """Return one row of inputs per position, ready to be given to the network."""
    positions = list(positions)
    inputs = np.zeros((len(positions), INPUT_SIZE), dtype=np.uint8)
    for row, position in enumerate(positions):
        encode_into(position, inputs[row])
    return inputs
