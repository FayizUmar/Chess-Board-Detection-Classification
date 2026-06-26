"""Strategy C: board-frame / outer-quad anchored detection.

Where :mod:`so101_nexus_core.chess_vision.detect_sb` anchors on *interior*
corners (which chess pieces frequently occlude) and
:mod:`so101_nexus_core.chess_vision.detect_hough` reconstructs the grid from
many short line segments, this strategy anchors on the board's *outer
boundary* -- the single large quadrilateral enclosing the whole playing area.

For a board photographed close to top-down, that boundary survives heavy piece
occlusion (pieces sit *inside* it) and yields a clean four-point contour. We
extract candidate quadrilaterals from several Canny edge maps, map each
directly onto the 800x800 canvas, and keep whichever placement scores best.
Because a quad has a four-fold rotational ambiguity (any of its corners could
be the board's top-left), all four rotations are tried and disambiguated by
:func:`~so101_nexus_core.chess_vision.scoring.combined_score`, whose
periodicity term only rewards a placement that lands the 8x8 grid coherently on
the canvas.

This complements the other two strategies: it is the reliable path for
near-top-down boards whose interior corners are buried under pieces, while
contributing only low-scoring (harmless) candidates for steeply-angled or
partially-off-frame boards whose outer boundary is itself broken -- the
pipeline's periodicity gate discards those.
"""

from __future__ import annotations

import cv2
import numpy as np

from so101_nexus_core.chess_vision.constants import WARP_SIZE
from so101_nexus_core.chess_vision.geometry import order_quad
from so101_nexus_core.chess_vision.scoring import combined_score

# Canvas corners in [TL, TR, BR, BL] order, matching geometry.order_quad.
_CANVAS = np.array(
    [[0, 0], [WARP_SIZE, 0], [WARP_SIZE, WARP_SIZE], [0, WARP_SIZE]], dtype=np.float32
)

# A board boundary should enclose a meaningful fraction of the frame but never
# (nearly) the whole image -- the latter is the desk / room outline, not a board.
_MIN_AREA_FRAC = 0.04
_MAX_AREA_FRAC = 0.92


def _candidate_quads(gray: np.ndarray) -> list[np.ndarray]:
    """Extract candidate board-boundary quadrilaterals from a grayscale image.

    Runs several Canny thresholds (board boundary contrast varies with board
    material and lighting), and for each contour large enough to plausibly be
    the board, tries a range of ``approxPolyDP`` epsilons until it collapses to
    a convex four-point polygon.
    """
    quads: list[np.ndarray] = []
    area = gray.shape[0] * gray.shape[1]
    for lo, hi in ((50, 150), (30, 100), (80, 200)):
        edges = cv2.Canny(gray, lo, hi)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            a = cv2.contourArea(c)
            if a < _MIN_AREA_FRAC * area or a > _MAX_AREA_FRAC * area:
                continue
            peri = cv2.arcLength(c, True)
            for frac in np.linspace(0.01, 0.06, 10):
                approx = cv2.approxPolyDP(c, frac * peri, True)
                if len(approx) == 4 and cv2.isContourConvex(approx):
                    quads.append(order_quad(approx.reshape(4, 2)))
                    break
    return quads


def _dedup(quads: list[np.ndarray]) -> list[np.ndarray]:
    """Drop near-duplicate quads (corners agreeing to within ~8 px)."""
    seen: set[tuple[int, ...]] = set()
    unique: list[np.ndarray] = []
    for q in quads:
        key = tuple(np.round(q.ravel() / 8).astype(int))
        if key not in seen:
            seen.add(key)
            unique.append(q)
    return unique


def try_frame(
    gray: np.ndarray, img_bgr: np.ndarray, mask: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray, str] | None:
    """Detect the board via its outer boundary quadrilateral.

    Parameters
    ----------
    gray : np.ndarray
        Working-resolution grayscale image.
    img_bgr : np.ndarray
        The matching BGR image, warped on success.
    mask : np.ndarray
        Board-region mask (same pixel space), passed to
        :func:`~so101_nexus_core.chess_vision.scoring.combined_score` for the
        geometric sanity term.

    Returns
    -------
    tuple of (score, H, warped_bgr, label), or None
        ``None`` if no plausible board-boundary quadrilateral was found.
    """
    best: tuple[float, np.ndarray, np.ndarray, str] | None = None
    for quad in _dedup(_candidate_quads(gray)):
        for rot in range(4):
            src = np.roll(quad, rot, axis=0).astype(np.float32)
            h_mat = cv2.getPerspectiveTransform(src, _CANVAS)
            warped = cv2.warpPerspective(img_bgr, h_mat, (WARP_SIZE, WARP_SIZE))
            score = combined_score(warped, h_mat, mask)
            if best is None or score > best[0]:
                best = (score, h_mat, warped, "FRAME")
    return best
