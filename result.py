"""Result types returned by the chess-vision pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

    from so101_nexus_core.chess_vision.engine import EngineMove
    from so101_nexus_core.chess_vision.pieces import PieceDetection


@dataclass
class Stage1Result:
    """Outcome of running board detection and perspective correction on one image.

    Attributes
    ----------
    warped : np.ndarray
        ``(800, 800, 3)`` BGR top-down board image.
    H : np.ndarray
        3x3 float64 homography mapping full-resolution source -> warped.
    debug : np.ndarray
        Source image with the detected grid projected back onto it.
    success : bool
        ``True`` if a board was found.
    score : float
        Checkerboard quality metric (higher is better).
    method : str
        Human-readable description of the pipeline path taken.
    """

    warped: np.ndarray
    H: np.ndarray
    debug: np.ndarray
    success: bool
    score: float
    method: str


@dataclass
class Stage2Result:
    """Outcome of running piece detection on one image's Stage 1 result.

    Attributes
    ----------
    detections : list of PieceDetection
        Raw YOLO detections on the original (unwarped) source image.
    board : dict
        Algebraic square (e.g. ``"e1"``) to FEN piece letter (e.g. ``"K"``).
    fen : str
        Full FEN string built from ``board`` via
        :func:`~so101_nexus_core.chess_vision.pieces.squares_to_fen`.
    debug : np.ndarray
        Source image with bounding boxes and bottom-center markers drawn.
    """

    detections: list[PieceDetection]
    board: dict[str, str]
    fen: str
    debug: np.ndarray


@dataclass
class Stage3Result:
    """Outcome of running Stockfish analysis on one image's Stage 2 FEN.

    Attributes
    ----------
    move : EngineMove
        Stockfish's recommended move.
    debug : np.ndarray
        Grid overlay with the move's source/destination squares
        translucently highlighted.
    """

    move: EngineMove
    debug: np.ndarray
