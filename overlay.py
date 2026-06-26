"""Debug and visualisation drawing on top of warped/source board images."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

from so101_nexus_core.chess_vision.constants import SQ, WARP_SIZE

if TYPE_CHECKING:
    from so101_nexus_core.chess_vision.engine import EngineMove
    from so101_nexus_core.chess_vision.pieces import PieceDetection


def draw_grid_overlay(warped: np.ndarray) -> np.ndarray:
    """Draw the 8x8 grid with algebraic labels on a warped board image."""
    out = warped.copy()
    for i in range(9):
        cv2.line(out, (i * SQ, 0), (i * SQ, WARP_SIZE), (0, 255, 0), 2, cv2.LINE_AA)
        cv2.line(out, (0, i * SQ), (WARP_SIZE, i * SQ), (0, 255, 0), 2, cv2.LINE_AA)
    files = "abcdefgh"
    ranks = "87654321"
    for col in range(8):
        for row in range(8):
            label = f"{files[col]}{ranks[row]}"
            cx = col * SQ + SQ // 2 - 10
            cy = row * SQ + SQ // 2 + 5
            cv2.putText(
                out, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3, cv2.LINE_AA
            )
            cv2.putText(
                out, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA
            )
    return out


def draw_source_debug(img: np.ndarray, h_mat: np.ndarray) -> np.ndarray:
    """Back-project the ideal 9x9 grid onto the source image using ``h_mat``'s inverse."""
    out = img.copy()
    h_inv = np.linalg.inv(h_mat)
    n_interp = 50  # interpolation points per line for smooth curves

    for i in range(9):
        # horizontal grid line
        pts_h = np.array([[t * SQ, i * SQ] for t in np.linspace(0, 8, n_interp)], dtype=np.float32)
        proj_h = cv2.perspectiveTransform(pts_h.reshape(-1, 1, 2), h_inv).reshape(-1, 2).astype(int)
        for k in range(len(proj_h) - 1):
            cv2.line(out, tuple(proj_h[k]), tuple(proj_h[k + 1]), (0, 255, 0), 1, cv2.LINE_AA)
        # vertical grid line
        pts_v = np.array([[i * SQ, t * SQ] for t in np.linspace(0, 8, n_interp)], dtype=np.float32)
        proj_v = cv2.perspectiveTransform(pts_v.reshape(-1, 1, 2), h_inv).reshape(-1, 2).astype(int)
        for k in range(len(proj_v) - 1):
            cv2.line(out, tuple(proj_v[k]), tuple(proj_v[k + 1]), (0, 255, 0), 1, cv2.LINE_AA)
    return out


def draw_piece_detections(img_bgr: np.ndarray, detections: list[PieceDetection]) -> np.ndarray:
    """Draw YOLO bounding boxes and a bottom-center marker on the source image.

    Operates on the original, unwarped image -- piece detection runs there,
    not on the Stage 1 ``warped`` output.
    """
    out = img_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = (round(v) for v in det.bbox)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2, cv2.LINE_AA)
        label = f"{det.class_name} {det.confidence:.2f}"
        label_pos = (x1, max(y1 - 6, 0))
        cv2.putText(out, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(
            out, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA
        )
        cx, cy = (round(v) for v in det.bottom_center)
        cv2.circle(out, (cx, cy), 8, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(out, (cx, cy), 6, (0, 0, 255), -1, cv2.LINE_AA)
    return out


def draw_move_highlight(
    warped_bgr: np.ndarray,
    move: EngineMove,
    from_color: tuple[int, int, int] = (0, 255, 255),
    to_color: tuple[int, int, int] = (0, 0, 255),
    alpha: float = 0.4,
) -> np.ndarray:
    """Translucently highlight a move's source and destination squares.

    Operates on the warped (axis-aligned, 800x800) board, the same space
    ``draw_grid_overlay`` uses, so squares are simple rects rather than
    perspective-skewed quads. Blended via ``cv2.addWeighted`` rather than
    drawn opaque so the piece sprite underneath stays visible.
    """
    files = "abcdefgh"
    ranks = "87654321"
    out = warped_bgr.copy()
    overlay = out.copy()
    for square, color in ((move.from_square, from_color), (move.to_square, to_color)):
        col = files.index(square[0])
        row = ranks.index(square[1])
        x0, y0 = col * SQ, row * SQ
        cv2.rectangle(overlay, (x0, y0), (x0 + SQ, y0 + SQ), color, -1)
    return cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)
