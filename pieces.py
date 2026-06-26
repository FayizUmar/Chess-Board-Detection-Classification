"""Chess piece detection (Stage 2): YOLO boxes -> board squares -> FEN.

Requires the ``chess-piece-detection`` extra (``ultralytics``); that
dependency is imported lazily inside :func:`load_model` so importing this
module (or the rest of :mod:`so101_nexus_core.chess_vision`) never requires
``ultralytics``/``torch`` unless piece detection is actually used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from so101_nexus_core.chess_vision.constants import SQ, WARP_SIZE

if TYPE_CHECKING:
    from pathlib import Path

# Class names emitted by the trained model, mapped to standard FEN letters
# (white pieces uppercase, black lowercase).
CLASS_TO_FEN: dict[str, str] = {
    "white-pawn": "P",
    "white-knight": "N",
    "white-bishop": "B",
    "white-rook": "R",
    "white-queen": "Q",
    "white-king": "K",
    "black-pawn": "p",
    "black-knight": "n",
    "black-bishop": "b",
    "black-rook": "r",
    "black-queen": "q",
    "black-king": "k",
}

_FILES = "abcdefgh"
_RANKS = "87654321"


@dataclass
class PieceDetection:
    """One YOLO detection on the original (unwarped) source image.

    Attributes
    ----------
    class_name : str
        Model class label, e.g. ``"white-pawn"``.
    confidence : float
        Detection confidence in ``[0, 1]``.
    bbox : tuple of float
        ``(x1, y1, x2, y2)`` in source-image pixel coordinates.
    bottom_center : tuple of float
        ``((x1 + x2) / 2, y2)`` -- the bottom-center of the box, used as the
        piece's board-plane contact point for homography projection.
    """

    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]
    bottom_center: tuple[float, float]


def load_model(weights_path: str | Path) -> Any:
    """Load a YOLO model from ``weights_path``.

    Isolated in its own function so the ``ultralytics`` import (which pulls
    in ``torch``) only happens when piece detection is actually requested.

    Importing ``torch`` sets ``OMP_NUM_THREADS=1``, which silently drops
    OpenCV's thread count via ``cv2.setNumThreads`` as a side effect. Running
    single-threaded changes ``cv2.findChessboardCornersSB``'s internal
    behaviour enough to turn correct Stage 1 detections into failures, so
    restore OpenCV's own thread count immediately after the import.
    """
    import cv2
    from ultralytics import YOLO

    model = YOLO(str(weights_path))
    cv2.setNumThreads(-1)
    return model


def detect_pieces(model: Any, img_bgr: np.ndarray, conf: float = 0.25) -> list[PieceDetection]:
    """Run YOLO inference on the original, full-resolution, unwarped image.

    Parameters
    ----------
    model : ultralytics.YOLO
        A model loaded via :func:`load_model`.
    img_bgr : np.ndarray
        The original source image (BGR), *not* the Stage 1 ``warped`` output
        -- pieces are tall and look correct only from their native angle.
    conf : float
        Minimum detection confidence to keep.

    Returns
    -------
    list of PieceDetection
    """
    results = model(img_bgr, conf=conf, verbose=False)
    names = results[0].names
    detections = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
        class_name = names[int(box.cls[0])]
        detections.append(
            PieceDetection(
                class_name=class_name,
                confidence=float(box.conf[0]),
                bbox=(x1, y1, x2, y2),
                bottom_center=((x1 + x2) / 2.0, y2),
            )
        )
    return detections


def assign_squares(
    detections: list[PieceDetection], h_mat: np.ndarray
) -> dict[str, PieceDetection]:
    """Map each detection's bottom-center through ``h_mat`` onto a board square.

    Uses the same ``files``/``ranks``/``SQ`` convention as
    :func:`so101_nexus_core.chess_vision.overlay.draw_grid_overlay`: column 0
    (canvas x near 0) is file ``a``, row 0 (canvas y near 0) is rank ``8``.
    Detections whose projected point falls outside the 800x800 canvas are
    off the board entirely (e.g. a false-positive detection on background
    clutter) and are dropped rather than forced onto an edge square. On a
    same-square collision among the remaining detections, the
    higher-confidence one wins.

    Parameters
    ----------
    detections : list of PieceDetection
        Detections in source-image pixel coordinates.
    h_mat : np.ndarray
        3x3 homography mapping source-image pixels to the 800x800 warped
        canvas (:attr:`so101_nexus_core.chess_vision.result.Stage1Result.H`).

    Returns
    -------
    dict mapping algebraic square (e.g. ``"e4"``) to the winning PieceDetection.
    """
    if not detections:
        return {}

    pts = np.array([d.bottom_center for d in detections], dtype=np.float32).reshape(-1, 1, 2)
    canvas_pts = cv2.perspectiveTransform(pts, h_mat).reshape(-1, 2)

    squares: dict[str, PieceDetection] = {}
    for detection, (cx, cy) in zip(detections, canvas_pts, strict=True):
        col = int(cx // SQ)
        row = int(cy // SQ)
        if not (0 <= col <= 7 and 0 <= row <= 7):
            continue
        square = f"{_FILES[col]}{_RANKS[row]}"
        existing = squares.get(square)
        if existing is None or detection.confidence > existing.confidence:
            squares[square] = detection
    return squares


def _flip_square(square: str) -> str:
    """Rotate an algebraic square 180 degrees (``a1`` <-> ``h8``, ``a8`` <-> ``h1``)."""
    file_, rank_ = square[0], square[1]
    return f"{_FILES[7 - _FILES.index(file_)]}{_RANKS[7 - _RANKS.index(rank_)]}"


def correct_orientation(
    warped: np.ndarray,
    h_mat: np.ndarray,
    square_map: dict[str, PieceDetection],
    *,
    min_pieces_per_color: int = 2,
    min_rank_margin: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, dict[str, PieceDetection]]:
    """Rotate the board 180 degrees if pieces landed on the wrong-colour ranks.

    Stage 1's warp has no notion of which side is White (see the module
    docs) -- it always labels the warped canvas's top-left corner ``a8``
    regardless of the photo's real-world orientation. A square-color check
    (e.g. "a1 is dark") *cannot* resolve this either: a standard board's
    coloring has 180-degree rotational symmetry (``a8``/``h1`` are always
    the same color, and so are ``a1``/``h8``), so corner intensity carries
    no information about which side is White. Piece positions are the only
    usable signal.

    Detect the mistake from where White's and Black's pieces actually ended
    up: White should average a lower rank number than Black. The average is
    confidence-weighted (``sum(rank * confidence) / sum(confidence)``) so a
    barely-confident misclassification -- e.g. a real photo's
    ``white-rook`` detection at 0.445 confidence -- can't outvote a
    confident one (0.87) on the same board. Two guard rails keep a weak
    signal from flipping a board that's actually fine: ``min_pieces_per_color``
    (need at least this many detections of *each* colour before trusting the
    comparison at all) and ``min_rank_margin`` (the weighted averages must
    differ by at least this much -- a near-tie is noise, not a confident
    orientation signal).

    Physically rotates ``warped``/``h_mat`` 180 degrees rather than just
    relabelling ``square_map`` -- the grid/move-highlight overlays draw
    directly on the warped canvas using the fixed top-left-is-``a8``
    convention, so a label-only fix would desync them from the corrected
    FEN.

    Parameters
    ----------
    warped : np.ndarray
        Stage 1's ``(800, 800, 3)`` BGR warped board image.
    h_mat : np.ndarray
        Stage 1's homography, in the same space as ``warped``.
    square_map : dict
        Output of :func:`assign_squares` for this image, still in Stage 1's
        unverified orientation.
    min_pieces_per_color : int
        Minimum detections of each colour required before the comparison is
        trusted at all.
    min_rank_margin : float
        Minimum difference between White's and Black's confidence-weighted
        average rank required to flip.

    Returns
    -------
    tuple of (warped, h_mat, square_map)
        Unchanged if there isn't enough evidence to flip (too few pieces of
        one colour, or the averages are too close), otherwise rotated
        180 degrees.
    """
    white = [
        (int(sq[1]), det.confidence)
        for sq, det in square_map.items()
        if det.class_name.startswith("white")
    ]
    black = [
        (int(sq[1]), det.confidence)
        for sq, det in square_map.items()
        if det.class_name.startswith("black")
    ]
    if len(white) < min_pieces_per_color or len(black) < min_pieces_per_color:
        return warped, h_mat, square_map

    white_avg = sum(rank * conf for rank, conf in white) / sum(conf for _, conf in white)
    black_avg = sum(rank * conf for rank, conf in black) / sum(conf for _, conf in black)
    if white_avg - black_avg < min_rank_margin:
        return warped, h_mat, square_map

    flipped_warped = cv2.rotate(warped, cv2.ROTATE_180)
    flip = np.array([[-1, 0, WARP_SIZE], [0, -1, WARP_SIZE], [0, 0, 1]], dtype=np.float64)
    flipped_h = flip @ h_mat
    flipped_map = {_flip_square(sq): det for sq, det in square_map.items()}
    return flipped_warped, flipped_h, flipped_map


def squares_to_fen(square_map: dict[str, str]) -> str:
    """Build a full FEN string from a square -> FEN-letter mapping.

    Parameters
    ----------
    square_map : dict
        Maps algebraic squares (e.g. ``"e4"``) to FEN piece letters
        (e.g. ``"K"``). Squares not present are treated as empty.

    Returns
    -------
    str
        Full FEN, e.g. the starting position
        ``"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"``.
        The trailing fields (active color, castling, en passant, halfmove,
        fullmove) are fixed defaults -- a single static image can't reveal
        whose turn it is or prior move history.
    """
    rows = []
    for rank in _RANKS:
        row_str = ""
        empty_run = 0
        for file_ in _FILES:
            letter = square_map.get(f"{file_}{rank}")
            if letter is None:
                empty_run += 1
            else:
                if empty_run:
                    row_str += str(empty_run)
                    empty_run = 0
                row_str += letter
        if empty_run:
            row_str += str(empty_run)
        rows.append(row_str)
    return "/".join(rows) + " w KQkq - 0 1"
