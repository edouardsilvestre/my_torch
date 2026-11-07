"""Chess board that can be played on, and detection of attacked squares and checks."""

from dataclasses import dataclass

from analyzer.fen import Position, parse_fen

WHITE = "w"
BLACK = "b"
FILES = "abcdefgh"
KNIGHT_STEPS = ((-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1))
KING_STEPS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
ROOK_DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))
BISHOP_DIRECTIONS = ((-1, -1), (-1, 1), (1, -1), (1, 1))


def square_index(row: int, file: int) -> int:
    """Index of a square: row 0 is rank 8 and file 0 is file a, as in the FEN."""
    return row * 8 + file


def square_name(index: int) -> str:
    return f"{FILES[index % 8]}{8 - index // 8}"


def parse_square(name: str) -> int:
    return square_index(8 - int(name[1]), FILES.index(name[0]))


def color_of(piece: str) -> str:
    return WHITE if piece.isupper() else BLACK


def opponent(color: str) -> str:
    return BLACK if color == WHITE else WHITE


def own(piece: str, color: str) -> str:
    """Return the letter of a piece for the given color, 'Q' or 'q' for a queen."""
    return piece.upper() if color == WHITE else piece.lower()


@dataclass
class Board:
    """Chess position that can be modified, with 64 squares indexed from a8 to h1."""

    squares: list[str | None]
    active_color: str
    castling: str
    en_passant: int | None
    halfmove_clock: int
    fullmove_number: int

    @classmethod
    def from_position(cls, position: Position) -> "Board":
        squares = [piece for rank in position.board for piece in rank]
        castling = "" if position.castling == "-" else position.castling
        en_passant = None if position.en_passant is None else parse_square(position.en_passant)
        return cls(squares, position.active_color, castling, en_passant,
                   position.halfmove_clock, position.fullmove_number)

    @classmethod
    def from_fen(cls, text: str) -> "Board":
        return cls.from_position(parse_fen(text))

    def copy(self) -> "Board":
        return Board(list(self.squares), self.active_color, self.castling, self.en_passant,
                     self.halfmove_clock, self.fullmove_number)

    def to_fen(self) -> str:
        ranks = []
        for row in range(8):
            text, empty = "", 0
            for piece in self.squares[row * 8:row * 8 + 8]:
                if piece is None:
                    empty += 1
                else:
                    text += (str(empty) if empty else "") + piece
                    empty = 0
            ranks.append(text + (str(empty) if empty else ""))
        castling = "".join(right for right in "KQkq" if right in self.castling) or "-"
        en_passant = "-" if self.en_passant is None else square_name(self.en_passant)
        return (f"{'/'.join(ranks)} {self.active_color} {castling} {en_passant} "
                f"{self.halfmove_clock} {self.fullmove_number}")

    def piece_at(self, row: int, file: int) -> str | None:
        if 0 <= row < 8 and 0 <= file < 8:
            return self.squares[square_index(row, file)]
        return None

    def is_attacked(self, square: int, by_color: str) -> bool:
        """Tell whether a piece of the given color attacks a square."""
        row, file = divmod(square, 8)
        pawn_row = row + 1 if by_color == WHITE else row - 1
        if any(self.piece_at(pawn_row, file + side) == own("p", by_color) for side in (-1, 1)):
            return True
        if any(self.piece_at(row + dr, file + df) == own("n", by_color)
               for dr, df in KNIGHT_STEPS):
            return True
        if any(self.piece_at(row + dr, file + df) == own("k", by_color)
               for dr, df in KING_STEPS):
            return True
        for directions, sliders in ((ROOK_DIRECTIONS, "rq"), (BISHOP_DIRECTIONS, "bq")):
            attackers = {own(piece, by_color) for piece in sliders}
            for dr, df in directions:
                r, f = row + dr, file + df
                while 0 <= r < 8 and 0 <= f < 8:
                    piece = self.squares[square_index(r, f)]
                    if piece is not None:
                        if piece in attackers:
                            return True
                        break
                    r, f = r + dr, f + df
        return False

    def king_square(self, color: str) -> int | None:
        king = own("k", color)
        return next((index for index, piece in enumerate(self.squares) if piece == king), None)

    def in_check(self, color: str) -> bool:
        """Tell whether the king of the given color is attacked."""
        square = self.king_square(color)
        return square is not None and self.is_attacked(square, opponent(color))
