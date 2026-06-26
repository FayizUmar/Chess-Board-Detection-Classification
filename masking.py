"""Colour-invariant texture segmentation isolating the chessboard from its background.

A chessboard's defining visual trait isn't a specific pair of colours -- it's
that adjacent squares alternate light/dark, producing strong local contrast in
a regular pattern. Detecting "busy, high local-contrast" regions via a local
standard-deviation map generalises across any board style (wood, black/white,
green/cream, marble, ...) without colour-specific tuning, unlike a fixed HSV
hue range.
"""

from __future__ import annotations

import cv2
import numpy as np


def board_mask(img_bgr: np.ndarray) -> np.ndarray:
    """Return a binary mask isolating the chessboard from a neutral background.

    Works for any board colour scheme: it thresholds local grayscale contrast
    rather than specific hues, since most backgrounds (carpet, walls,
    furniture) are comparatively flat next to a checkerboard's alternating
    squares.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    win = max(9, (min(img_bgr.shape[:2]) // 25) | 1)  # odd box-filter window
    mean = cv2.boxFilter(gray, -1, (win, win))
    mean_sq = cv2.boxFilter(gray * gray, -1, (win, win))
    std = np.sqrt(np.clip(mean_sq - mean * mean, 0, None))

    std_u8 = cv2.normalize(std, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(std_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=4)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=2)

    # Keep only the single largest blob
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean = np.zeros_like(mask)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        rect = cv2.minAreaRect(largest)
        rect_area = rect[1][0] * rect[1][1]
        rectangularity = area / rect_area if rect_area > 0 else 0.0
        if rectangularity < 0.75:
            # The blob likely bled into adjacent clutter (e.g. an occluding
            # arm or limb) via the morphological closing above and is no
            # longer board-shaped. Fall back to its bounding quadrilateral
            # rather than leaking the non-board area into downstream checks.
            box = cv2.boxPoints(rect).astype(np.int32)
            cv2.drawContours(clean, [box], -1, 255, -1)
        else:
            cv2.drawContours(clean, [largest], -1, 255, -1)
    return clean


def corner_density_mask(img_bgr: np.ndarray) -> np.ndarray:
    """Localise the board by the density of corner (X-junction) features.

    Complements :func:`board_mask`. The local-standard-deviation segmentation
    there keys on alternating light/dark *texture*, which fades to nothing on a
    board whose squares are large relative to the std window -- each square's
    flat interior reads as low contrast, so the board never lights up and the
    mask latches onto busier surroundings (carpet, clothing, furniture edges)
    instead. A checkerboard's most scale-invariant signature is different: the
    dense, regular lattice of saddle-point corners where four squares meet.
    Shi-Tomasi corner detection fires densely there regardless of square size,
    so blurring those corner hits into a density map and keeping the strongest
    blob localises the board even when contrast segmentation cannot.

    Returns the filled convex hull of the densest corner blob, dilated for a
    small safety margin. Returned in the same pixel space as the input. Note
    busy non-board texture (e.g. carpet) also produces corners, so this is a
    coarse region for restricting detection/scoring, not a board-tight outline.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=2000, qualityLevel=0.01, minDistance=8)
    if corners is None:
        return np.zeros((h, w), np.uint8)

    density = np.zeros((h, w), np.float32)
    for x, y in corners.reshape(-1, 2).astype(int):
        density[y, x] = 1.0
    density = cv2.GaussianBlur(density, (0, 0), 25)
    density_u8 = cv2.normalize(density, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, dens_mask = cv2.threshold(density_u8, 90, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(dens_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros((h, w), np.uint8)
    if contours:
        hull = cv2.convexHull(max(contours, key=cv2.contourArea))
        cv2.fillConvexPoly(mask, hull, 255)
        mask = cv2.dilate(mask, np.ones((25, 25), np.uint8), iterations=1)
    return mask
