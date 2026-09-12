"""Hand-checked unit tests for the pure-Python tokenizer.

Run with:  python3 test_tokenizer.py
(Works with plain assertions; pytest also collects these if available.)
"""

import numpy as np

from tokenizer import (
    ACTION_TO_MOVE,
    CHAR_TO_INDEX,
    CLASS_TOKEN,
    INPUT_VOCAB_SIZE,
    MOVE_TO_ACTION,
    NUM_ACTIONS,
    PAD_TOKEN,
    RECENT_MOVES_LENGTH,
    SEQUENCE_LENGTH,
    VOCAB,
    prepare_recent_moves_tokens,
    tokenize,
)

STARTPOS = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def test_constants():
    assert len(VOCAB) == 31
    assert INPUT_VOCAB_SIZE == 33
    assert CLASS_TOKEN == 31
    assert PAD_TOKEN == 32
    assert SEQUENCE_LENGTH == 78
    assert RECENT_MOVES_LENGTH == 12
    # 'w' and 'b' both resolve; 'b' shares the file-letter index 11.
    assert CHAR_TO_INDEX["w"] == 29
    assert CHAR_TO_INDEX["b"] == 11
    assert CHAR_TO_INDEX["."] == 30


def test_startpos_length_and_dtype():
    toks = tokenize(STARTPOS)
    assert toks.dtype == np.uint8
    assert len(toks) == SEQUENCE_LENGTH == 78


def test_startpos_first_token_is_side_to_move():
    toks = tokenize(STARTPOS)
    # First token is the side-to-move 'w'.
    assert toks[0] == CHAR_TO_INDEX["w"] == 29


def test_startpos_specific_indices():
    toks = tokenize(STARTPOS)
    # After 'w' comes the board: 'rnbqkbnr' then 'pppppppp' ...
    assert toks[1] == CHAR_TO_INDEX["r"]  # a8 rook
    assert toks[2] == CHAR_TO_INDEX["n"]  # b8 knight
    assert toks[3] == CHAR_TO_INDEX["b"]  # c8 bishop
    assert toks[4] == CHAR_TO_INDEX["q"]  # d8 queen
    assert toks[5] == CHAR_TO_INDEX["k"]  # e8 king
    # 64 board tokens follow the side token -> indices 1..64.
    # The final board rank is "RNBQKBNR" occupying indices 57..64.
    assert toks[57] == CHAR_TO_INDEX["R"]  # a1 rook
    assert toks[64] == CHAR_TO_INDEX["R"]  # h1 rook
    # Last token is always the class token.
    assert toks[-1] == CLASS_TOKEN


def test_empty_rows_expand_to_dots():
    # Startpos ranks 3-6 are '8' each -> 32 empty squares -> 32 dots.
    toks = tokenize(STARTPOS)
    board_tokens = toks[1:65]  # 64 board squares
    dot = CHAR_TO_INDEX["."]
    # Middle 32 squares (ranks 3-6) are all dots.
    assert list(board_tokens[16:48]) == [dot] * 32


def test_fully_empty_board_all_dots():
    toks = tokenize("8/8/8/8/8/8/8/8 w - - 0 1")
    dot = CHAR_TO_INDEX["."]
    assert list(toks[1:65]) == [dot] * 64


def test_castling_padding():
    dot = CHAR_TO_INDEX["."]
    # No castling rights -> 4 dots.
    toks_none = tokenize("8/8/8/8/8/8/8/8 w - - 0 1")
    # board is 1 (side) + 64 = 65 tokens; castling occupies indices 65..68.
    assert list(toks_none[65:69]) == [dot] * 4

    # Partial castling 'Kq' -> [K, q, ., .]
    toks_part = tokenize("8/8/8/8/8/8/8/8 w Kq - 0 1")
    assert list(toks_part[65:69]) == [
        CHAR_TO_INDEX["K"],
        CHAR_TO_INDEX["q"],
        dot,
        dot,
    ]

    # Full castling 'KQkq' -> exactly those 4, no padding.
    toks_full = tokenize(STARTPOS)
    assert list(toks_full[65:69]) == [
        CHAR_TO_INDEX["K"],
        CHAR_TO_INDEX["Q"],
        CHAR_TO_INDEX["k"],
        CHAR_TO_INDEX["q"],
    ]


def test_en_passant_padding():
    dot = CHAR_TO_INDEX["."]
    # No en passant -> 2 dots at indices 69..70.
    toks_none = tokenize(STARTPOS)
    assert list(toks_none[69:71]) == [dot, dot]

    # En passant square 'c6' -> ['c','6'] (no padding).
    toks_ep = tokenize("rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2")
    assert list(toks_ep[69:71]) == [CHAR_TO_INDEX["c"], CHAR_TO_INDEX["6"]]


def test_halfmove_fullmove_padding():
    dot = CHAR_TO_INDEX["."]
    # Startpos: halfmove '0' -> [0, ., .]; fullmove '1' -> [1, ., .].
    toks = tokenize(STARTPOS)
    # halfmove at 71..73, fullmove at 74..76, class token at 77.
    assert list(toks[71:74]) == [CHAR_TO_INDEX["0"], dot, dot]
    assert list(toks[74:77]) == [CHAR_TO_INDEX["1"], dot, dot]
    assert toks[77] == CLASS_TOKEN

    # Multi-digit clocks fill without padding when 3 digits.
    toks2 = tokenize("8/8/8/8/8/8/8/8 w - - 99 150")
    assert list(toks2[71:74]) == [CHAR_TO_INDEX["9"], CHAR_TO_INDEX["9"], dot]
    assert list(toks2[74:77]) == [
        CHAR_TO_INDEX["1"],
        CHAR_TO_INDEX["5"],
        CHAR_TO_INDEX["0"],
    ]


def test_tokenize_rejects_short_fen():
    try:
        tokenize("8/8/8/8/8/8/8/8 w - -")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on short FEN")


def test_action_space_size_and_uniqueness():
    assert NUM_ACTIONS == 1968
    assert len(MOVE_TO_ACTION) == NUM_ACTIONS
    assert len(ACTION_TO_MOVE) == NUM_ACTIONS
    # Round trip.
    for action, move in ACTION_TO_MOVE.items():
        assert MOVE_TO_ACTION[move] == action
    # A couple of sanity members.
    assert "a1a2" in MOVE_TO_ACTION  # rook/queen up the a-file
    assert "b1c3" in MOVE_TO_ACTION  # knight move
    assert "e7e8q" in MOVE_TO_ACTION  # straight promotion to queen
    assert "a7b8n" in MOVE_TO_ACTION  # capture promotion to knight


def test_prepare_recent_moves_left_pads():
    arr = prepare_recent_moves_tokens(["e2e4", "e7e5"])
    assert arr.dtype == np.int64
    assert len(arr) == RECENT_MOVES_LENGTH == 12
    # Left-padded with PAD_TOKEN; the two moves are right-aligned.
    assert list(arr[:10]) == [PAD_TOKEN] * 10
    assert arr[10] == MOVE_TO_ACTION["e2e4"]
    assert arr[11] == MOVE_TO_ACTION["e7e5"]


def test_prepare_recent_moves_truncates_to_last_12():
    moves = [ACTION_TO_MOVE[i] for i in range(20)]  # 20 valid moves
    arr = prepare_recent_moves_tokens(moves)
    assert len(arr) == 12
    # Should keep only the last 12 moves, in order.
    expected = [MOVE_TO_ACTION[m] for m in moves[-12:]]
    assert list(arr) == expected


def test_prepare_recent_moves_empty():
    arr = prepare_recent_moves_tokens([])
    assert list(arr) == [PAD_TOKEN] * 12


def test_prepare_recent_moves_rejects_bad_move():
    try:
        prepare_recent_moves_tokens(["zzzz"])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on invalid move")


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        t()
        print(f"  PASS {t.__name__}")
        passed += 1
    print(f"\n{passed}/{len(tests)} tokenizer tests passed.")


if __name__ == "__main__":
    _run_all()
