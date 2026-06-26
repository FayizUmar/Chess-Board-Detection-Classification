"""Strategy B: Hough-line grid detection.

Canny + HoughLinesP within the colour-segmented board region. Lines are
clustered into two roughly-perpendicular families, the best 9 per family are
selected, and their 81 intersections feed a RANSAC homography.
"""

from __future__ import annotations

import cv2
import numpy as np

from so101_nexus_core.chess_vision.constants import SQ, WARP_SIZE
from so101_nexus_core.chess_vision.geometry import (
    angle_dist,
    cluster_by_position,
    intersect,
    select_consecutive_windows,
)
from so101_nexus_core.chess_vision.scoring import combined_score


def try_hough(
    gray_pp: np.ndarray, img_bgr: np.ndarray, mask: np.ndarray, cluster_gap: float = 25.0
) -> tuple[float, np.ndarray, np.ndarray, str] | None:
    """Detect grid lines and fit a RANSAC homography to their intersections.

    Parameters
    ----------
    gray_pp, img_bgr, mask
        Preprocessed grayscale, matching BGR image, and board-region mask.
    cluster_gap : float
        Perpendicular-position gap (px) passed to
        :func:`~so101_nexus_core.chess_vision.geometry.cluster_by_position`.
        The default suits boards whose grid lines are >~ one gap apart; a
        smaller value is needed for near-overhead boards foreshortened so much
        that adjacent grid lines fall within the default gap and would
        otherwise merge into too few clusters. The pipeline runs both.

    Returns
    -------
    tuple of (score, H, warped_bgr, label), or None
        ``None`` if not enough lines/clusters/intersections were found.
    """
    # Edge detection within the board mask
    masked = cv2.bitwise_and(gray_pp, gray_pp, mask=mask)
    edges = cv2.Canny(masked, 40, 120)
    edges = cv2.dilate(edges, np.ones((2, 2)), iterations=1)

    segs = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180, threshold=40, minLineLength=60, maxLineGap=15
    )
    if segs is None or len(segs) < 18:
        return None
    lines = [s[0] for s in segs]

    # classify line angles
    angles = [np.degrees(np.arctan2(line[3] - line[1], line[2] - line[0])) % 180 for line in lines]
    hist, _ = np.histogram(angles, bins=180, range=(0, 180))
    hist_smooth = np.convolve(hist, np.ones(5) / 5, mode="same")
    peak1 = int(np.argmax(hist_smooth))
    suppressed = hist_smooth.copy()
    # Angle is periodic mod 180, so suppress circularly around peak1: a board
    # shot near-overhead can have its dominant family at ~2deg, whose other
    # half sits at ~178deg. Plain (non-wrapping) slicing would leave that half
    # un-suppressed and let peak2 latch onto the *same* direction, yielding two
    # parallel families with no grid intersections.
    for off in range(-25, 26):
        suppressed[(peak1 + off) % 180] = 0
    peak2 = int(np.argmax(suppressed))

    tol = 15
    fam_a = [line for line, a in zip(lines, angles, strict=True) if angle_dist(a, peak1) < tol]
    fam_b = [line for line, a in zip(lines, angles, strict=True) if angle_dist(a, peak2) < tol]
    if len(fam_a) < 5 or len(fam_b) < 5:
        return None

    cl_a = cluster_by_position(fam_a, peak1, cluster_gap)
    cl_b = cluster_by_position(fam_b, peak2, cluster_gap)
    if len(cl_a) < 7 or len(cl_b) < 7:
        return None

    # Spacing uniformity alone can't tell a real grid line from the board's
    # outer wooden frame edge (often itself ~one square-width from the last
    # grid line). Try multiple candidate windows per axis and let checkerboard
    # contrast on the resulting warp disambiguate, mirroring try_sb's
    # exhaustive-offset scoring.
    windows_a = select_consecutive_windows(cl_a)
    windows_b = select_consecutive_windows(cl_b)

    img_h, img_w = img_bgr.shape[:2]
    best = None
    for sel_a in windows_a:
        for sel_b in windows_b:
            na, nb = len(sel_a), len(sel_b)
            src_pts, dst_pts = [], []
            for i, (_, _, line_a) in enumerate(sel_a):
                for j, (_, _, line_b) in enumerate(sel_b):
                    pt = intersect(line_a, line_b)
                    if pt and 0 <= pt[0] < img_w and 0 <= pt[1] < img_h:
                        src_pts.append(pt)
                        dst_pts.append([j * SQ, i * SQ])
            if len(src_pts) < 4:
                continue

            h_mat, inlier_mask = cv2.findHomography(
                np.array(src_pts, dtype=np.float32),
                np.array(dst_pts, dtype=np.float32),
                cv2.RANSAC,
                5.0,
            )
            if h_mat is None:
                continue
            warped = cv2.warpPerspective(img_bgr, h_mat, (WARP_SIZE, WARP_SIZE))
            score = combined_score(warped, h_mat, mask)
            n_inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
            if best is None or score > best[0]:
                best = (score, h_mat, warped, f"Hough({na}x{nb},{n_inliers}inl)")
    return best
