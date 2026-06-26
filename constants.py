"""Tunable size constants shared across the chess-vision pipeline."""

from __future__ import annotations

WORK_LONG_EDGE = 1024
"""Resize the longest edge of an input image to this many pixels before processing."""

WARP_SIZE = 800
"""Output side length, in pixels, of the top-down warped board."""

SQ = WARP_SIZE // 8
"""Pixel width of one chess square in the warped output (100 px)."""

MIN_PERIODICITY = 0.3
"""Minimum :func:`~so101_nexus_core.chess_vision.scoring.periodicity_score` for
a detection to count as success.

Correctly aligned boards score ~0.75-0.9; geometrically wrong warps score
< 0.01, so this threshold sits in a wide empty gap. Below it the pipeline reports
failure (and writes a ``_debug_FAIL`` image) rather than emitting a confidently
wrong grid."""

SB_ATTEMPTS = 4
"""Times to retry ``cv2.findChessboardCornersSB`` per pattern size.

The SB detector is internally non-deterministic and intermittently returns a
false negative (~10% per call) on borderline / partially-occluded boards. Each
retry is an independent draw, so a handful of attempts makes a detectable
pattern reliably found and keeps the pipeline's chosen pattern stable run to
run."""
