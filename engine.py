"""Chess engine integration (Stage 3): FEN -> Stockfish -> best move.

Requires the ``chess-engine`` extra (``chess``, i.e. python-chess) plus a
real Stockfish binary installed separately (e.g. ``brew install stockfish``)
-- python-chess only speaks the UCI protocol to an existing binary, it does
not ship one. Both the ``chess`` import and the binary lookup are deferred to
:func:`load_engine`/:func:`best_move` so importing this module never requires
either unless engine analysis is actually used.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import chess.engine


@dataclass
class EngineMove:
    """Stockfish's recommended move for one position.

    Attributes
    ----------
    uci : str
        Move in UCI notation, e.g. ``"e2e4"``.
    san : str
        Move in standard algebraic notation, e.g. ``"e4"``.
    from_square : str
        Source algebraic square, e.g. ``"e2"``.
    to_square : str
        Destination algebraic square, e.g. ``"e4"``.
    """

    uci: str
    san: str
    from_square: str
    to_square: str


def _apply_turn(fen: str, turn: str) -> str:
    """Return ``fen`` with its active-color field replaced by ``turn``.

    ``squares_to_fen()`` always emits ``"w"`` as a placeholder (a single
    photo can't reveal whose move it is); the caller supplies the real
    answer here. Pure string manipulation -- no ``chess`` import needed, so
    it's unit-testable without python-chess or Stockfish installed.
    """
    fields = fen.split(" ")
    fields[1] = turn
    return " ".join(fields)


def load_engine(stockfish_path: str | None = None) -> chess.engine.SimpleEngine:
    """Launch a Stockfish UCI subprocess and return the engine handle.

    Isolated in its own function so the ``chess`` import (and the search for
    an actual Stockfish binary) only happens when engine analysis is
    requested. Caller is responsible for calling ``.quit()`` on the result.

    Parameters
    ----------
    stockfish_path : str, optional
        Explicit path to the Stockfish binary. If omitted, looks it up on
        ``PATH`` via ``shutil.which("stockfish")``.

    Raises
    ------
    RuntimeError
        If ``stockfish_path`` doesn't point to an existing file, or none was
        given and none is found on ``PATH``.
    """
    resolved = stockfish_path or shutil.which("stockfish")
    if not resolved or not os.path.isfile(resolved):
        raise RuntimeError(
            f"Stockfish binary not found{f' at {resolved!r}' if resolved else ''}. "
            "Install it separately (e.g. `brew install stockfish` on macOS, "
            "`apt install stockfish` on Linux) or pass its path explicitly."
        )

    import chess.engine

    return chess.engine.SimpleEngine.popen_uci(resolved)


def best_move(engine: Any, fen: str, turn: str = "w", time_limit: float = 1.0) -> EngineMove:
    """Ask Stockfish for its recommended move in the position described by ``fen``.

    Parameters
    ----------
    engine : chess.engine.SimpleEngine
        A live engine handle from :func:`load_engine`.
    fen : str
        Full FEN string (e.g. from :func:`~so101_nexus_core.chess_vision.pieces.squares_to_fen`).
    turn : str
        ``"w"`` or ``"b"`` -- overrides ``fen``'s active-color field, since
        Stage 2 can't determine whose move it is from a single photo.
    time_limit : float
        Seconds Stockfish is given to search.

    Returns
    -------
    EngineMove

    Raises
    ------
    ValueError
        If the position (after applying ``turn``) is illegal (e.g. a
        missing king from a bad piece detection) or already game-over.
    """
    import chess
    import chess.engine

    fixed_fen = _apply_turn(fen, turn)
    try:
        board = chess.Board(fixed_fen)
    except ValueError as exc:
        raise ValueError(f"Invalid board position, cannot analyze: {exc}") from exc

    # squares_to_fen() always claims full "KQkq" rights since a single photo
    # can't reveal move history; drop whatever rights the current piece
    # layout can't actually support (e.g. a king that isn't on its home
    # square) rather than rejecting an otherwise-legal position over a
    # placeholder mismatch. This can't recover rights lost earlier in a game
    # where the king/rook happen to be back home, but it's strictly closer
    # to the truth than blindly claiming all four.
    board.castling_rights = board.clean_castling_rights()

    # chess.Board() only validates FEN *syntax*; a wrong king count (e.g.
    # from a YOLO misdetection) parses fine but crashes the Stockfish
    # subprocess outright once handed to engine.play(), so check legality
    # explicitly first.
    if not board.is_valid():
        raise ValueError(
            f"Detected position is not a legal chess position ({board.status()!r}); "
            "likely a piece-detection error"
        )

    if board.is_game_over():
        raise ValueError(f"Position is already game-over ({board.outcome()}), no move to compute")

    result = engine.play(board, chess.engine.Limit(time=time_limit))
    move = result.move
    return EngineMove(
        uci=move.uci(),
        san=board.san(move),
        from_square=chess.square_name(move.from_square),
        to_square=chess.square_name(move.to_square),
    )
