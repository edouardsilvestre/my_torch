"""Labels of chess positions, computed with the rules of chess."""

from dataset.board import Board, opponent
from dataset.moves import has_legal_move

COLOR_NAMES = {"w": "White", "b": "Black"}


def label_position(board: Board) -> str | None:
    """Return the label of a position, or None when the position is illegal.

    The color of a label is the one of the side that gives the check or the mate, which
    is the side that does not have to move. A position where the side that does not have
    to move is itself in check cannot happen in a game: it is illegal.
    """
    mover = board.active_color
    attacker = opponent(mover)
    if board.in_check(attacker):
        return None
    if not board.in_check(mover):
        return "Nothing"
    state = "Check" if has_legal_move(board) else "Checkmate"
    return f"{state} {COLOR_NAMES[attacker]}"
