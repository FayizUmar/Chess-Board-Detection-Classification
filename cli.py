"""Command-line entry point for the chess-vision Stage 1 pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2

from so101_nexus_core.chess_vision.engine import best_move, load_engine
from so101_nexus_core.chess_vision.overlay import (
    draw_grid_overlay,
    draw_move_highlight,
    draw_piece_detections,
)
from so101_nexus_core.chess_vision.pieces import (
    CLASS_TO_FEN,
    assign_squares,
    correct_orientation,
    detect_pieces,
    load_model,
    squares_to_fen,
)
from so101_nexus_core.chess_vision.pipeline import detect_and_warp


def _run_stage2(img: Any, result: Any, model: Any) -> tuple[str, Any]:
    detections = detect_pieces(model, img)
    square_to_detection = assign_squares(detections, result.H)
    result.warped, result.H, square_to_detection = correct_orientation(
        result.warped, result.H, square_to_detection
    )
    board = {square: CLASS_TO_FEN[det.class_name] for square, det in square_to_detection.items()}
    fen = squares_to_fen(board)
    return fen, draw_piece_detections(img, detections)


def _run_stage3(engine: Any, fen: str, result: Any, args: argparse.Namespace) -> Any | None:
    try:
        move = best_move(engine, fen, turn=args.turn, time_limit=args.engine_time)
    except ValueError as exc:
        print(f"  Stockfish FAIL: {exc}")
        return None
    print(f"  Move: {move.uci} ({move.san})")
    return draw_move_highlight(draw_grid_overlay(result.warped), move)


def _process_image(
    path: Path, outdir: Path, model: Any, engine: Any, args: argparse.Namespace
) -> None:
    img = cv2.imread(str(path))
    if img is None:
        print(f"FAIL {path.name}: could not read image")
        return

    result = detect_and_warp(img)
    stem = path.stem

    if not result.success:
        print(f"FAIL {path.name}: no board detected")
        cv2.imwrite(str(outdir / f"{stem}_debug_FAIL.jpg"), result.debug)
        return

    print(f"OK {path.name}: score={result.score:.2f} method={result.method}")
    cv2.imwrite(str(outdir / f"{stem}_debug.jpg"), result.debug)

    if model is None:
        cv2.imwrite(str(outdir / f"{stem}_warped.jpg"), result.warped)
        cv2.imwrite(str(outdir / f"{stem}_grid.jpg"), draw_grid_overlay(result.warped))
        return

    fen, pieces_debug = _run_stage2(img, result, model)
    print(f"  FEN: {fen}")
    # _run_stage2 may have rotated result.warped 180 degrees (see
    # correct_orientation), so write the warped/grid debug images only now
    # -- otherwise they'd show Stage 1's unverified, possibly-backwards
    # orientation instead of the one the FEN actually describes.
    cv2.imwrite(str(outdir / f"{stem}_warped.jpg"), result.warped)
    cv2.imwrite(str(outdir / f"{stem}_grid.jpg"), draw_grid_overlay(result.warped))
    cv2.imwrite(str(outdir / f"{stem}_pieces.jpg"), pieces_debug)

    if engine is None:
        return

    highlight = _run_stage3(engine, fen, result, args)
    if highlight is not None:
        cv2.imwrite(str(outdir / f"{stem}_move.jpg"), highlight)


def main() -> None:
    """Run board detection (and, with --model/--stockfish, Stage 2/3) on one or more images."""
    parser = argparse.ArgumentParser(description="Chess board Stage 1: detect & warp")
    parser.add_argument("images", nargs="+", help="Input image path(s)")
    parser.add_argument("-o", "--outdir", default="./output", help="Output directory")
    parser.add_argument(
        "--model",
        default=None,
        help="Path to a YOLO piece-detection .pt file (requires the "
        "chess-piece-detection extra). If omitted, only Stage 1 board "
        "detection runs.",
    )
    parser.add_argument(
        "--stockfish",
        nargs="?",
        const="",
        default=None,
        metavar="PATH",
        help="Enable Stage 3 move calculation. Optionally pass a path to the "
        "stockfish binary; if omitted, it's looked up on PATH. Requires the "
        "chess-engine extra and a Stockfish binary, and --model (Stage 3 "
        "needs Stage 2's FEN).",
    )
    parser.add_argument(
        "--turn",
        choices=["w", "b"],
        default="w",
        help="Side to move, for Stage 3 analysis (default: w). A static photo "
        "can't reveal whose turn it is.",
    )
    parser.add_argument(
        "--engine-time",
        type=float,
        default=1.0,
        help="Seconds Stockfish is given to search per position (default: 1.0).",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.model) if args.model else None

    engine = None
    if args.stockfish is not None:
        try:
            engine = load_engine(args.stockfish or None)
        except RuntimeError as exc:
            print(f"FAIL: {exc}")
            return

    try:
        for image_path in args.images:
            _process_image(Path(image_path), outdir, model, engine, args)
    finally:
        if engine is not None:
            engine.quit()
