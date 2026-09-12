"""Pure-Python tokenizer and UCI action space for the ChessMimic clock model.

This is an ORIGINAL, dependency-free reimplementation of the tokenization
scheme used by the ChessMimic project (https://github.com/thomasj02/1e4_ai).
It produces token arrays and a move/action mapping that are byte-for-byte
compatible with the upstream C++ binding (``chessmimic_core``) and the
upstream searchless-chess-derived Python reference, so a model trained with
the upstream tooling can be served without building any C++ extension.

The tokenizer intentionally has NO dependency on ``python-chess``: the queen
and knight geometry needed to build the action space is computed directly.

Nothing in this file is copied from the upstream C++ sources; it is a clean
reimplementation from the documented scheme.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Vocabulary and constants
# ---------------------------------------------------------------------------

# Index == position in this list (31 characters).
VOCAB: List[str] = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "a", "b", "c", "d", "e", "f", "g", "h",
    "p", "n", "r", "k", "q",
    "P", "B", "N", "R", "Q", "K",
    "w", ".",
]

CHAR_TO_INDEX: Dict[str, int] = {ch: i for i, ch in enumerate(VOCAB)}

# Derived constants (must match the upstream scheme exactly).
INPUT_VOCAB_SIZE = len(VOCAB) + 2  # + class token + pad token = 33
CLASS_TOKEN = len(VOCAB)           # 31
PAD_TOKEN = len(VOCAB) + 1         # 32
SEQUENCE_LENGTH = 78               # 77 board/meta tokens + 1 class token

# Board-run digits that expand to that many empty squares (".").
SPACES_CHARACTERS = frozenset({"1", "2", "3", "4", "5", "6", "7", "8"})

# Number of recent moves fed to the model.
RECENT_MOVES_LENGTH = 12

_DOT = CHAR_TO_INDEX["."]


def tokenize(fen: str) -> np.ndarray:
    """Tokenize a FEN string into a ``uint8`` numpy array of length 78.

    Build order:
      1. side-to-move token
      2. board squares (digits expand to that many '.' tokens)
      3. castling (padded to 4 with '.'; '-' => 4 dots)
      4. en passant ('-' => 2 dots; otherwise the square's chars, no padding)
      5. halfmove clock, padded to 3 with '.'
      6. fullmove number, padded to 3 with '.'
      7. class token
    """
    parts = fen.split(" ")
    if len(parts) < 6:
        raise ValueError(
            f"FEN must have 6 space-separated fields, got {len(parts)}: {fen!r}"
        )
    board, side, castling, en_passant, halfmove, fullmove = parts[:6]

    board = board.replace("/", "")

    indices: List[int] = []

    # 1. Side to move ('w' or 'b'). Both exist in the vocab ('b' is a file
    #    letter), so a direct lookup is correct.
    indices.append(CHAR_TO_INDEX[side])

    # 2. Board squares.
    for ch in board:
        if ch in SPACES_CHARACTERS:
            indices.extend(int(ch) * [_DOT])
        else:
            indices.append(CHAR_TO_INDEX[ch])

    # 3. Castling rights, always 4 tokens wide.
    if castling == "-":
        indices.extend(4 * [_DOT])
    else:
        for ch in castling:
            indices.append(CHAR_TO_INDEX[ch])
        if len(castling) < 4:
            indices.extend((4 - len(castling)) * [_DOT])

    # 4. En passant square, 2 tokens (or 2 dots).
    if en_passant == "-":
        indices.extend(2 * [_DOT])
    else:
        for ch in en_passant:
            indices.append(CHAR_TO_INDEX[ch])

    # 5. Halfmove clock, padded to 3.
    halfmove_padded = halfmove + "." * (3 - len(halfmove))
    indices.extend(CHAR_TO_INDEX[ch] for ch in halfmove_padded)

    # 6. Fullmove number, padded to 3.
    fullmove_padded = fullmove + "." * (3 - len(fullmove))
    indices.extend(CHAR_TO_INDEX[ch] for ch in fullmove_padded)

    # 7. Class token.
    indices.append(CLASS_TOKEN)

    if len(indices) != SEQUENCE_LENGTH:
        raise ValueError(
            f"Tokenized length {len(indices)} != {SEQUENCE_LENGTH} for FEN {fen!r}"
        )

    return np.asarray(indices, dtype=np.uint8)


# ---------------------------------------------------------------------------
# UCI action space
# ---------------------------------------------------------------------------

_FILES = ["a", "b", "c", "d", "e", "f", "g", "h"]


def _square_index(file_idx: int, rank_idx: int) -> int:
    """Square index 0..63 with a1=0, b1=1, ... h8=63 (python-chess convention)."""
    return rank_idx * 8 + file_idx


def _square_name(square: int) -> str:
    return _FILES[square % 8] + str(square // 8 + 1)


def _queen_targets(square: int) -> List[int]:
    """Squares a queen attacks from ``square`` on an empty board.

    Returned in ascending square-index order, matching the iteration used by
    the upstream generator (python-chess ``SquareSet`` / the C++ bitboard
    scan both walk squares 0..63 ascending).
    """
    file_idx = square % 8
    rank_idx = square // 8
    targets: List[int] = []
    directions = [
        (1, 0), (-1, 0), (0, 1), (0, -1),      # rook lines
        (1, 1), (1, -1), (-1, 1), (-1, -1),    # bishop lines
    ]
    for df, dr in directions:
        f, r = file_idx + df, rank_idx + dr
        while 0 <= f < 8 and 0 <= r < 8:
            targets.append(_square_index(f, r))
            f += df
            r += dr
    targets.sort()
    return targets


def _knight_targets(square: int) -> List[int]:
    """Squares a knight attacks from ``square``, ascending square-index order."""
    file_idx = square % 8
    rank_idx = square // 8
    targets: List[int] = []
    offsets = [
        (1, 2), (2, 1), (2, -1), (1, -2),
        (-1, -2), (-2, -1), (-2, 1), (-1, 2),
    ]
    for df, dr in offsets:
        f, r = file_idx + df, rank_idx + dr
        if 0 <= f < 8 and 0 <= r < 8:
            targets.append(_square_index(f, r))
    targets.sort()
    return targets


def _compute_all_possible_actions() -> Tuple[Dict[str, int], Dict[int, str]]:
    """Build the full UCI move/action mapping.

    Order (must match upstream):
      * regular moves: for each from-square 0..63, queen attacks (ascending
        target square) followed by knight attacks (ascending target square)
      * promotion moves: for ranks (2->1) then (7->8), for each file a..h,
        straight then left-capture then right-capture, each with promotions
        in the order q, r, b, n.
    """
    all_moves: List[str] = []

    # Regular (non-promotion) moves.
    for square in range(64):
        from_name = _square_name(square)
        for target in _queen_targets(square):
            all_moves.append(from_name + _square_name(target))
        for target in _knight_targets(square):
            all_moves.append(from_name + _square_name(target))

    # Promotion moves.
    promotion_moves: List[str] = []
    for rank, next_rank in [("2", "1"), ("7", "8")]:
        for index_file, file in enumerate(_FILES):
            # Straight promotion.
            base = f"{file}{rank}{file}{next_rank}"
            promotion_moves += [base + piece for piece in ("q", "r", "b", "n")]
            # Left capture.
            if index_file > 0:
                next_file = _FILES[index_file - 1]
                base = f"{file}{rank}{next_file}{next_rank}"
                promotion_moves += [base + piece for piece in ("q", "r", "b", "n")]
            # Right capture.
            if index_file < len(_FILES) - 1:
                next_file = _FILES[index_file + 1]
                base = f"{file}{rank}{next_file}{next_rank}"
                promotion_moves += [base + piece for piece in ("q", "r", "b", "n")]
    all_moves += promotion_moves

    move_to_action: Dict[str, int] = {}
    action_to_move: Dict[int, str] = {}
    for action, move in enumerate(all_moves):
        assert move not in move_to_action, f"duplicate move {move}"
        move_to_action[move] = action
        action_to_move[action] = move
    return move_to_action, action_to_move


MOVE_TO_ACTION, ACTION_TO_MOVE = _compute_all_possible_actions()
NUM_ACTIONS = len(MOVE_TO_ACTION)


def prepare_recent_moves_tokens(recent_moves: List[str]) -> np.ndarray:
    """Map recent UCI moves to an ``int64`` array of length 12.

    Takes the last ``RECENT_MOVES_LENGTH`` moves and LEFT-pads with
    ``PAD_TOKEN`` so the most recent move is right-aligned, matching the
    upstream C++ ``prepare_recent_moves_tokens``.
    """
    if recent_moves is None:
        recent_moves = []

    tokens = [PAD_TOKEN] * RECENT_MOVES_LENGTH
    trimmed = list(recent_moves)[-RECENT_MOVES_LENGTH:]
    offset = RECENT_MOVES_LENGTH - len(trimmed)
    for i, move in enumerate(trimmed):
        if move not in MOVE_TO_ACTION:
            raise ValueError(f"Invalid move: {move}")
        tokens[offset + i] = MOVE_TO_ACTION[move]

    return np.asarray(tokens, dtype=np.int64)
