"""Quality metrics used to rank candidate board warps.

Each function here is a pure NumPy/OpenCV computation over an already-warped
image array, with no detection logic, so they can be unit tested against
synthetic checkerboard arrays without running any detector.
"""

from __future__ import annotations

import cv2
import numpy as np

from so101_nexus_core.chess_vision.constants import SQ, WARP_SIZE


def checkerboard_score(gray: np.ndarray) -> float:
    """Mean absolute intensity difference between adjacent cells of an 8x8 grid.

    Computed over every pair of horizontally/vertically adjacent cells in an
    800x800 warped grayscale image. A well-aligned warp maximises this because
    adjacent squares alternate dark <-> light.
    """
    margin = SQ // 5
    total, count = 0.0, 0
    for r in range(8):
        for c in range(8):
            cell = gray[
                r * SQ + margin : (r + 1) * SQ - margin,
                c * SQ + margin : (c + 1) * SQ - margin,
            ]
            if cell.size == 0:
                continue
            mean_val = float(np.mean(cell))
            for dr, dc in ((0, 1), (1, 0)):
                nr, nc = r + dr, c + dc
                if nr < 8 and nc < 8:
                    neighbour = gray[
                        nr * SQ + margin : (nr + 1) * SQ - margin,
                        nc * SQ + margin : (nc + 1) * SQ - margin,
                    ]
                    if neighbour.size > 0:
                        total += abs(float(np.mean(neighbour)) - mean_val)
                        count += 1
    return total / max(count, 1)


def _cell_means(gray: np.ndarray, margin: int | None = None) -> np.ndarray:
    """Mean intensity of each of the 64 squares on an 800x800 warped grid.

    Cells with zero area after margin-trimming (not reachable in practice
    with the module's fixed ``SQ``/margin, but guarded for robustness)
    default to ``0.0``.
    """
    if margin is None:
        margin = SQ // 5
    means = np.zeros((8, 8), dtype=np.float64)
    for r in range(8):
        for c in range(8):
            cell = gray[
                r * SQ + margin : (r + 1) * SQ - margin,
                c * SQ + margin : (c + 1) * SQ - margin,
            ]
            if cell.size:
                means[r, c] = float(np.mean(cell))
    return means


def periodicity_score(gray: np.ndarray) -> float:
    """Fraction of the 8x8 cell-mean variance explained by the checkerboard mode.

    Builds the 8x8 matrix of per-cell mean intensities, then projects it onto
    the ideal checkerboard basis ``(-1)**(r+c)`` (the highest-frequency 2D mode)
    and returns that mode's share of the matrix's total variance.

    Unlike :func:`checkerboard_score`, which only sums *local* adjacent-cell
    contrast, this measures whether the *whole* 800x800 canvas is one coherent
    8x8 checkerboard. A geometrically wrong / oversized warp -- whose grid spills
    onto a robot arm, desk, or legs -- has high local contrast in places but does
    not alternate consistently across the board, so it scores near 0 here while a
    correctly aligned board scores ~0.75-0.9. This cleanly rejects the wrong warps
    that ``checkerboard_score`` alone cannot tell apart from real boards.

    Note this is *phase-blind*: it squares the checkerboard-mode coefficient,
    so it cannot distinguish a correctly-oriented board from one rotated 90
    degrees (which alternates just as cleanly, only with light/dark swapped).
    See :func:`correct_square_color_orientation` for the check that does
    care about phase.
    """
    means = _cell_means(gray)

    means -= means.mean()
    total = float(np.sum(means * means))
    if total <= 0:
        return 0.0

    basis_1d = np.array([1, -1, 1, -1, 1, -1, 1, -1], dtype=np.float64)
    board = np.outer(basis_1d, basis_1d)
    coef = float(np.sum(means * board)) / 64.0  # board has 64 unit-magnitude entries
    energy = coef * coef * 64.0
    return energy / total


def correct_square_color_orientation(
    warped: np.ndarray, h_mat: np.ndarray, min_contrast: float = 10.0
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate the warp 90 degrees if its square colors contradict the FIDE rule.

    This is a *different* orientation fix from
    :func:`so101_nexus_core.chess_vision.pieces.correct_orientation`, and the
    two are not interchangeable. Under the ``draw_grid_overlay`` convention
    (top-left of the canvas is ``a8``), the standard "a1 is dark" rule means
    the ``(row+col) % 2 == 0`` cells (``a8``, ``h1``) should be the *light*
    squares and the other parity group (``h8``, ``a1``) should be *dark*. A
    180-degree rotation of the canvas preserves ``(row+col) % 2`` for every
    cell, so it can never fix or cause a violation of that rule -- that's
    why square color is mathematically blind to the White/Black ambiguity
    ``pieces.correct_orientation`` handles, and why that function has to use
    piece positions instead. A 90-or-270-degree rotation (a file/rank
    transpose), on the other hand, *always* flips ``(row+col) % 2`` for
    every cell, which is exactly the kind of error this function catches:
    ``detect_frame``'s corner-ordering search picks among rotated candidates
    using ``periodicity_score``, which squares its coefficient and so can't
    tell a 90-degree-rotated board from a correctly-oriented one (both
    alternate equally cleanly) -- only the actual light/dark identity does.

    Always rotates 90 degrees clockwise when triggered, never 270 -- both
    invert parity identically, and either choice leaves at most a residual
    180-degree ambiguity for ``pieces.correct_orientation`` to resolve
    afterward from piece positions, so it doesn't matter which one is
    picked.

    Parameters
    ----------
    warped : np.ndarray
        Stage 1's ``(800, 800, 3)`` BGR warped board image.
    h_mat : np.ndarray
        Stage 1's homography, in the same space as ``warped``.
    min_contrast : float
        Minimum mean-intensity gap (0-255 scale) the wrong-side parity group
        must exceed the right-side one by before rotating. Guards against
        guessing on a low-contrast board where the signal isn't trustworthy
        (in which case this is a no-op, same as the "already correct" case).

    Returns
    -------
    tuple of (warped, h_mat)
        Unchanged unless the squares are confidently inverted, in which case
        rotated 90 degrees clockwise.
    """
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    means = _cell_means(gray)
    parity0 = [means[r, c] for r in range(8) for c in range(8) if (r + c) % 2 == 0]
    parity1 = [means[r, c] for r in range(8) for c in range(8) if (r + c) % 2 == 1]
    p0, p1 = float(np.mean(parity0)), float(np.mean(parity1))
    if p1 - p0 < min_contrast:
        return warped, h_mat

    turn = np.array([[0, -1, WARP_SIZE], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
    return cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE), turn @ h_mat


def board_coverage(gray: np.ndarray) -> float:
    """Fraction of the 800x800 output that contains actual board content.

    I.e. is not black padding introduced by the perspective warp.
    """
    return float(np.mean(gray > 15))


def mask_coverage(h: np.ndarray, mask: np.ndarray) -> float:
    """Fraction of the 800x800 canvas that maps back onto the source board mask.

    ``h`` is the candidate source-image -> warped-canvas homography and
    ``mask`` is a binary board-region mask in the *same* (un-warped) pixel
    space ``h`` maps from. A homography that over-extrapolates the board
    boundary into background (e.g. an occluding arm or desk) pulls non-board
    pixels into the canvas, lowering this fraction.
    """
    warped_mask = cv2.warpPerspective(mask, h, (WARP_SIZE, WARP_SIZE))
    return float(np.mean(warped_mask > 0))


def combined_score(
    warped_bgr: np.ndarray,
    h: np.ndarray | None = None,
    mask: np.ndarray | None = None,
) -> float:
    """Overall quality score used to rank candidate warps.

    Heavily penalises warps where the board doesn't fill the canvas. If
    ``h`` and ``mask`` are given, also penalises warps whose canvas extends
    beyond the segmented board region (see :func:`mask_coverage`). Always folds
    in :func:`periodicity_score`, which collapses geometrically wrong warps
    (grid spilling off the board) toward zero.
    """
    g = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
    cb = checkerboard_score(g)
    cov = board_coverage(g)
    pp = periodicity_score(g)
    mc = mask_coverage(h, mask) if h is not None and mask is not None else 1.0
    return cb * cov * mc * pp if cov > 0.45 else cb * 0.1
