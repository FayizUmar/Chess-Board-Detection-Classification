"""Strategy A: OpenCV's sector-based chessboard corner detector.

Works perfectly when >= 7x7 inner corners are visible (no piece occlusion).
For partial patterns (3x3 .. 6x6), an exhaustive grid-offset search tests
every possible placement within the 7x7 grid and scores the resulting warp
by checkerboard contrast.
"""

from __future__ import annotations

import cv2
import numpy as np

from so101_nexus_core.chess_vision.constants import SB_ATTEMPTS, SQ, WARP_SIZE
from so101_nexus_core.chess_vision.scoring import combined_score


def find_chessboard_sb(
    gray: np.ndarray, pattern: tuple[int, int], attempts: int = SB_ATTEMPTS
) -> tuple[bool, np.ndarray | None]:
    """Run ``cv2.findChessboardCornersSB`` with retries on false negatives.

    The SB detector intermittently returns ``False`` for a pattern it can
    actually find (~10% per call on borderline boards). Each call is an
    independent draw, so retrying makes a detectable pattern reliably found and
    stops the pipeline from flickering between pattern sizes run to run.
    """
    for _ in range(attempts):
        ok, corners = cv2.findChessboardCornersSB(gray, pattern)
        if ok:
            return True, corners
    return False, None


def try_sb(
    gray: np.ndarray, img_bgr: np.ndarray, mask: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray, str] | None:
    """Try SB corner detection with the largest pattern that succeeds.

    For partial patterns, exhaustively search all possible grid offsets and
    return the single best warp.

    Returns
    -------
    tuple of (score, H, warped_bgr, label), or None
        ``None`` if no pattern was detected.
    """
    best = None
    for pattern in ((7, 7), (6, 6), (5, 5), (4, 4), (3, 3)):
        ok, corners = find_chessboard_sb(gray, pattern)
        if not ok:
            continue
        n, m = pattern

        # Full 7x7 -> direct homography, no offset ambiguity
        if n == 7 and m == 7:
            dst = np.array(
                [[(j + 1) * SQ, (i + 1) * SQ] for i in range(7) for j in range(7)], dtype=np.float32
            )
            h_mat, _ = cv2.findHomography(corners, dst, cv2.RANSAC, 3.0)
            if h_mat is None:
                break
            w = cv2.warpPerspective(img_bgr, h_mat, (WARP_SIZE, WARP_SIZE))
            return (combined_score(w, h_mat, mask), h_mat, w, f"SB({n}x{m})")

        # Partial pattern -> try every valid offset within the 7x7 grid
        c2d = corners.reshape(n, m, 2)
        for row_off in range(8 - n):
            for col_off in range(8 - m):
                src_pts, dst_pts = [], []
                for i in range(n):
                    for j in range(m):
                        src_pts.append(c2d[i, j])
                        dst_pts.append([(col_off + j + 1) * SQ, (row_off + i + 1) * SQ])
                h_mat, _ = cv2.findHomography(
                    np.array(src_pts, dtype=np.float32),
                    np.array(dst_pts, dtype=np.float32),
                    cv2.RANSAC,
                    3.0,
                )
                if h_mat is None:
                    continue
                w = cv2.warpPerspective(img_bgr, h_mat, (WARP_SIZE, WARP_SIZE))
                sc = combined_score(w, h_mat, mask)
                if best is None or sc > best[0]:
                    best = (sc, h_mat, w, f"SB({n}x{m})@({row_off},{col_off})")
        break  # only use the largest detected pattern
    return best
