"""Chess board detection (Stage 1), piece detection (Stage 2), and engine analysis (Stage 3).

Stage 1 detects a wooden chessboard in an angled camera image and warps it
to an 800x800 top-down canonical view. Requires the ``chess-vision`` extra
(``pip install so101-nexus-core[chess-vision]``).

Stage 2 detects pieces with a YOLO model and maps them to algebraic squares
and a FEN string. Requires the additional ``chess-piece-detection`` extra
(``pip install so101-nexus-core[chess-piece-detection]``); importing this
package does not require it -- ``ultralytics`` is only imported inside
:func:`~so101_nexus_core.chess_vision.pieces.load_model`.

Stage 3 feeds that FEN to a Stockfish binary (via the ``chess`` package,
i.e. python-chess) to compute the recommended move. Requires the additional
``chess-engine`` extra (``pip install so101-nexus-core[chess-engine]``) plus
a Stockfish binary installed separately (e.g. ``brew install stockfish``);
importing this package requires neither -- ``chess`` is only imported inside
:func:`~so101_nexus_core.chess_vision.engine.load_engine`/:func:`~so101_nexus_core.chess_vision.engine.best_move`.
"""

from __future__ import annotations

import cv2

from so101_nexus_core.chess_vision.constants import SQ, WARP_SIZE
from so101_nexus_core.chess_vision.engine import EngineMove, best_move, load_engine
from so101_nexus_core.chess_vision.overlay import (
    draw_grid_overlay,
    draw_move_highlight,
    draw_piece_detections,
)
from so101_nexus_core.chess_vision.pieces import (
    CLASS_TO_FEN,
    PieceDetection,
    assign_squares,
    detect_pieces,
    load_model,
    squares_to_fen,
)
from so101_nexus_core.chess_vision.pipeline import detect_and_warp
from so101_nexus_core.chess_vision.result import Stage1Result, Stage2Result, Stage3Result

# macOS's OpenCL/Metal backend crashes inside findChessboardCornersSB
# (NSInvalidArgumentException in AppleMetalOpenGLRenderer); force CPU path.
cv2.ocl.setUseOpenCL(False)

__all__ = [
    "CLASS_TO_FEN",
    "SQ",
    "WARP_SIZE",
    "EngineMove",
    "PieceDetection",
    "Stage1Result",
    "Stage2Result",
    "Stage3Result",
    "assign_squares",
    "best_move",
    "detect_and_warp",
    "detect_pieces",
    "draw_grid_overlay",
    "draw_move_highlight",
    "draw_piece_detections",
    "load_engine",
    "load_model",
    "squares_to_fen",
]
