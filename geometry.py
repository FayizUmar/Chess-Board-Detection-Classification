"""Pure 2-D line geometry used by the Hough-line grid detector.

None of these functions touch OpenCV or pixel data -- they operate on plain
``(x1, y1, x2, y2)`` line-segment coordinates -- so they can be unit tested
with synthetic coordinates alone.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

Line = Sequence[float]
"""A line segment as ``(x1, y1, x2, y2)``."""

Cluster = tuple[float, float, Line]
"""A clustered group of parallel segments: ``(avg_position, total_length, longest_segment)``."""


def angle_dist(a: float, ref: float) -> float:
    """Return the minimum angular distance between ``a`` and ``ref`` on the [0, 180) circle."""
    return min(abs(a - ref), 180.0 - abs(a - ref))


def intersect(line1: Line, line2: Line) -> tuple[float, float] | None:
    """Return the intersection point of two infinite lines through ``line1`` and ``line2``.

    Parameters
    ----------
    line1, line2 : Line
        Line segments as ``(x1, y1, x2, y2)``; only their direction and one
        point are used, so the segments need not actually overlap.

    Returns
    -------
    tuple of float, or None
        The ``(x, y)`` intersection point, or ``None`` if the lines are
        (numerically) parallel.
    """
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-10:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))


def order_quad(pts: np.ndarray) -> np.ndarray:
    """Order four corner points as ``[top-left, top-right, bottom-right, bottom-left]``.

    Uses the standard coordinate-sum / coordinate-difference trick: the
    top-left corner has the smallest ``x + y`` and the bottom-right the
    largest, while the top-right has the smallest ``y - x`` and the
    bottom-left the largest. This gives a consistent, clockwise winding for
    any convex quadrilateral, which downstream homography code relies on to
    map corners to the warped canvas without flipping or rotating the board.

    Parameters
    ----------
    pts : np.ndarray
        Array of shape ``(4, 2)`` of ``(x, y)`` corner coordinates, in any order.

    Returns
    -------
    np.ndarray
        The same four points, shape ``(4, 2)``, dtype ``float32``, ordered
        ``[TL, TR, BR, BL]``.
    """
    p = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = p.sum(axis=1)
    # np.diff over axis=1 yields (col1 - col0) = y - x per row, so the
    # smallest is the top-right corner and the largest the bottom-left.
    d = np.diff(p, axis=1).ravel()
    return np.array(
        [p[np.argmin(s)], p[np.argmin(d)], p[np.argmax(s)], p[np.argmax(d)]],
        dtype=np.float32,
    )


def cluster_by_position(fam: Sequence[Line], ref_angle: float, gap: float = 25) -> list[Cluster]:
    """Group near-parallel line segments into clusters by perpendicular position.

    Segments are projected onto the axis perpendicular to ``ref_angle`` and
    sorted; consecutive segments closer than ``gap`` along that axis are
    merged into the same cluster (e.g. multiple short Hough segments that lie
    on the same grid line).

    Parameters
    ----------
    fam : sequence of Line
        Line segments belonging to one angular family.
    ref_angle : float
        Representative angle (degrees) of the family.
    gap : float
        Maximum perpendicular-position gap, in pixels, for two segments to be
        merged into the same cluster.

    Returns
    -------
    list of Cluster
        One ``(avg_position, total_length, longest_segment)`` tuple per cluster,
        ordered by position.
    """

    def sort_key(line: Line) -> float:
        mx = (line[0] + line[2]) / 2.0
        my = (line[1] + line[3]) / 2.0
        perp = np.radians(ref_angle + 90)
        return mx * np.cos(perp) + my * np.sin(perp)

    keyed = sorted([(sort_key(line), line) for line in fam], key=lambda x: x[0])
    clusters: list[list[tuple[float, Line]]] = []
    cur = [keyed[0]]
    for pos, line in keyed[1:]:
        if pos - cur[-1][0] < gap:
            cur.append((pos, line))
        else:
            clusters.append(cur)
            cur = [(pos, line)]
    clusters.append(cur)

    result: list[Cluster] = []
    for c in clusters:
        avg_pos = float(np.mean([p for p, _ in c]))
        longest = max(c, key=lambda x: np.hypot(x[1][2] - x[1][0], x[1][3] - x[1][1]))[1]
        total_len = sum(np.hypot(seg[2] - seg[0], seg[3] - seg[1]) for _, seg in c)
        result.append((avg_pos, float(total_len), longest))
    return result


def select_consecutive_windows(
    clusters: Sequence[Cluster], n: int = 9, top_k: int = 3
) -> list[list[Cluster]]:
    """Return up to ``top_k`` candidate ``n``-wide windows, ranked by spacing uniformity.

    A correctly-detected chessboard grid line family has ``n`` evenly spaced
    clusters. When more than ``n`` clusters were found (extra spurious lines --
    e.g. the board's outer wooden frame edge, which is often itself roughly one
    square-width from the last real grid line and so looks "evenly spaced"
    too), this slides an ``n``-wide window over them and ranks every window by
    its inter-cluster spacing variance and linear-trend residual.

    Spacing uniformity alone can't distinguish a frame edge from a real grid
    line, so the caller is expected to disambiguate between the returned
    candidates using image content (e.g. checkerboard contrast of the
    resulting warp) rather than relying on the single best-by-spacing window.

    Parameters
    ----------
    clusters : sequence of Cluster
        Clusters ordered by position, as returned by :func:`cluster_by_position`.
    n : int
        Number of consecutive clusters per window.
    top_k : int
        Maximum number of ranked candidate windows to return.

    Returns
    -------
    list of list of Cluster
        Up to ``top_k`` windows, best (lowest cost) first. A single window
        (all of ``clusters``) if there are ``<= n``.
    """
    if len(clusters) <= n:
        return [list(clusters)]

    ranked: list[tuple[float, list[Cluster]]] = []
    for start in range(len(clusters) - (n - 1)):
        sel = list(clusters[start : start + n])
        spacings = np.diff([x[0] for x in sel])
        cv_val = np.std(spacings) / (np.mean(spacings) + 1e-6)
        coeffs = np.polyfit(np.arange(len(spacings)), spacings, 1)
        residual = np.sum((spacings - np.polyval(coeffs, np.arange(len(spacings)))) ** 2)
        cost = residual + cv_val * 100
        ranked.append((cost, sel))
    ranked.sort(key=lambda x: x[0])
    return [sel for _, sel in ranked[:top_k]]
