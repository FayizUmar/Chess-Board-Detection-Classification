"""Chess Vision Pipeline -- Stage 1: Board Detection & Perspective Correction.

Detects a wooden chessboard in an angled camera image and warps it to an
800x800 top-down canonical view suitable for per-square piece classification.

Tested on natural-wood boards (brown/tan squares, Staunton pieces) with
camera angles from ~45 degrees to near-overhead, including slight rotation.

Strategy
--------
Three detection backends run in parallel, each optionally refined, and the
single best result is kept:

  A) :func:`so101_nexus_core.chess_vision.detect_sb.try_sb` --
     OpenCV's sector-based detector. Works perfectly when >= 7x7 inner
     corners are visible (no piece occlusion). For partial patterns
     (3x3 .. 6x6), an exhaustive grid-offset search tests every possible
     placement within the 7x7 grid and scores the resulting warp by
     checkerboard contrast.

  B) :func:`so101_nexus_core.chess_vision.detect_hough.try_hough` --
     Canny + HoughLinesP within the colour-segmented board region. Lines
     are clustered into two roughly-perpendicular families, the best 9 per
     family are selected, and their 81 intersections feed a RANSAC
     homography.

  C) :func:`so101_nexus_core.chess_vision.detect_frame.try_frame` --
     Anchors on the board's outer boundary quadrilateral rather than its
     interior corners, so it survives heavy piece occlusion on near-top-down
     boards. Candidate four-point contours are mapped directly onto the
     canvas and disambiguated (corner ordering and grid registration) by the
     periodicity term of :func:`combined_score`.

  Refinement -- :func:`so101_nexus_core.chess_vision.refine.refine_warp`
     re-runs SB on the (now nearly top-down) warped image after each coarse
     warp. Because perspective distortion is removed, SB usually finds a
     much larger pattern, yielding a high-precision correction homography
     composed with the coarse one.
"""

from __future__ import annotations

import cv2
import numpy as np

from so101_nexus_core.chess_vision.constants import MIN_PERIODICITY, WARP_SIZE, WORK_LONG_EDGE
from so101_nexus_core.chess_vision.detect_frame import try_frame
from so101_nexus_core.chess_vision.detect_hough import try_hough
from so101_nexus_core.chess_vision.detect_sb import try_sb
from so101_nexus_core.chess_vision.masking import board_mask, corner_density_mask
from so101_nexus_core.chess_vision.overlay import draw_source_debug
from so101_nexus_core.chess_vision.refine import refine_warp
from so101_nexus_core.chess_vision.result import Stage1Result
from so101_nexus_core.chess_vision.scoring import (
    combined_score,
    correct_square_color_orientation,
    periodicity_score,
)


def detect_and_warp(img_bgr: np.ndarray) -> Stage1Result:
    """Detect the chessboard and produce an 800x800 top-down warp.

    Parameters
    ----------
    img_bgr : np.ndarray
        Input image in BGR colour order (as returned by ``cv2.imread``), any
        resolution.

    Returns
    -------
    Stage1Result
        See :class:`so101_nexus_core.chess_vision.result.Stage1Result`.
    """
    # resize for processing
    h0, w0 = img_bgr.shape[:2]
    scale = WORK_LONG_EDGE / max(h0, w0)
    img_work = cv2.resize(img_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img_work, cv2.COLOR_BGR2GRAY)

    # preprocessed grayscale for Hough
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray_pp = cv2.bilateralFilter(clahe.apply(gray), 9, 75, 75)

    # board-region mask
    mask = board_mask(img_work)
    mask_inner = cv2.erode(mask, np.ones((5, 5)), iterations=2)

    # Pass 1: run all detection strategies
    sb_result = try_sb(gray, img_work, mask)
    hough_result = try_hough(gray_pp, img_work, mask_inner)
    frame_result = try_frame(gray, img_work, mask)

    results = [sb_result, hough_result, frame_result]

    # A board shot near-overhead foreshortens its grid so much that adjacent
    # lines fall within Hough's default cluster gap and merge into too few
    # clusters, so the default-gap pass above bails. Re-run Hough with a smaller
    # gap to recover those grids.
    results.append(try_hough(gray_pp, img_work, mask_inner, cluster_gap=12.0))

    # The same boards (large, flat-interior squares) also defeat the
    # contrast-based board_mask: each square reads as low contrast, so the std
    # mask under-fires on the board and bleeds into surrounding clutter.
    # corner_density_mask localises the board from its corner lattice instead;
    # run the small-gap Hough against it too, for the cases where the std mask
    # is the thing that failed rather than the gap.
    cd_mask = corner_density_mask(img_work)
    if cd_mask.any():
        cd_mask_inner = cv2.erode(cd_mask, np.ones((5, 5)), iterations=2)
        results.append(try_hough(gray_pp, img_work, cd_mask_inner, cluster_gap=12.0))

    # All of the above are extra candidates ranked by the same combined_score;
    # because that score multiplies in periodicity, an added candidate can only
    # outrank an existing one by being more genuinely board-like, so this cannot
    # regress a board already detected well, only rescue one that was not.

    # Pass 2: refine each, collect all candidates
    candidates = []
    for result in results:
        if result is None:
            continue
        score, h_coarse, warped_coarse, label = result
        candidates.append((score, h_coarse, warped_coarse, label))

        h_ref, w_ref, ref_suffix = refine_warp(img_work, h_coarse, warped_coarse, mask)
        if ref_suffix:
            ref_score = combined_score(w_ref, h_ref, mask)
            candidates.append((ref_score, h_ref, w_ref, label + ref_suffix))

    # pick the overall best
    if not candidates:
        return Stage1Result(
            warped=np.zeros((WARP_SIZE, WARP_SIZE, 3), np.uint8),
            H=np.eye(3),
            debug=img_work,
            success=False,
            score=0.0,
            method="NONE",
        )

    candidates.sort(key=lambda c: c[0], reverse=True)
    best_score, h_best, warped_best, method = candidates[0]

    # A geometrically wrong warp (grid spilling off the board) can still be the
    # best of a bad bunch; require real board periodicity to call it a success
    # rather than emit a confidently wrong grid.
    best_gray = cv2.cvtColor(warped_best, cv2.COLOR_BGR2GRAY)
    success = periodicity_score(best_gray) >= MIN_PERIODICITY

    # periodicity_score (above) is phase-blind and can't tell a 90-degree-
    # rotated candidate from a correctly-oriented one; fix that here, before
    # composing the work-resolution scale, so the rotation lives purely in
    # canvas space.
    warped_best, h_best = correct_square_color_orientation(warped_best, h_best)

    # h_best maps work-resolution pixels -> canvas; full-res pixels map to
    # work-resolution pixels via work_pt = scale * full_pt, so compose with
    # that scaling (not its inverse) to get full-res source -> canvas.
    s = np.diag([scale, scale, 1.0])
    h_full_res = h_best @ s

    # debug overlay on the working-resolution source
    debug_img = draw_source_debug(img_work, h_best)

    return Stage1Result(
        warped=warped_best,
        H=h_full_res,
        debug=debug_img,
        success=success,
        score=best_score,
        method=method,
    )
