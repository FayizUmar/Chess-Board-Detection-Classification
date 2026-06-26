"""Pass 2: post-warp SB refinement.

Re-runs the SB detector on the coarse-warped image (now nearly top-down).
Because perspective distortion is mostly removed, SB typically finds a
larger pattern than it did on the original photo, yielding a high-precision
correction homography composed with the coarse one.
"""

from __future__ import annotations

import cv2
import numpy as np

from so101_nexus_core.chess_vision.constants import SQ, WARP_SIZE
from so101_nexus_core.chess_vision.detect_sb import find_chessboard_sb
from so101_nexus_core.chess_vision.scoring import combined_score


def refine_warp(
    img_bgr: np.ndarray, h_coarse: np.ndarray, warped_coarse: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray, str]:
    """Re-run SB on the coarse-warped image and refine the homography if possible.

    All valid grid offsets are tested and the best correction homography is
    composed with ``h_coarse``.

    Returns
    -------
    tuple of (H_refined, warped_refined, label_suffix)
        ``label_suffix`` is ``""`` if no improvement was found.
    """
    gray_w = cv2.cvtColor(warped_coarse, cv2.COLOR_BGR2GRAY)

    for pattern in ((7, 7), (6, 6), (5, 5), (4, 4)):
        ok, corners = find_chessboard_sb(gray_w, pattern)
        if not ok:
            continue
        n, m = pattern

        best_score = -1.0
        best_h: np.ndarray | None = None
        best_warped: np.ndarray | None = None
        for row_off in range(8 - n):
            for col_off in range(8 - m):
                ideal = np.array(
                    [
                        [(col_off + j + 1) * SQ, (row_off + i + 1) * SQ]
                        for i in range(n)
                        for j in range(m)
                    ],
                    dtype=np.float32,
                )
                h_fix, _ = cv2.findHomography(corners, ideal, cv2.RANSAC, 3.0)
                if h_fix is None:
                    continue
                h_total = h_fix @ h_coarse
                w = cv2.warpPerspective(img_bgr, h_total, (WARP_SIZE, WARP_SIZE))
                sc = combined_score(w, h_total, mask)
                if sc > best_score:
                    best_score, best_h, best_warped = sc, h_total, w

        if best_h is not None:
            assert best_warped is not None
            return best_h, best_warped, f"+refine({n}x{m})"
        break  # pattern detected but no valid offset improved things

    return h_coarse, warped_coarse, ""
