"""Standalone piece detection -- Stage 2 only, no board warp / FEN / engine.

Runs a trained YOLO model directly on an image and writes an annotated copy.
Deliberately decoupled from the Stage 1/3 pipeline: it only imports ultralytics.

Usage:
    python run_piece_detection.py IMAGE [IMAGE ...] \
        [--model chess-model-yolov8m.pt] [-o ./output] [--conf 0.25]
"""

from __future__ import annotations

import argparse
from pathlib import Path

_DEFAULT_MODEL = Path(__file__).with_name("chess-model-yolov8m.pt")


def main() -> None:
    parser = argparse.ArgumentParser(description="Piece detection only (YOLO, no board/FEN).")
    parser.add_argument("images", nargs="+", help="Input image path(s).")
    parser.add_argument(
        "--model",
        default=str(_DEFAULT_MODEL),
        help=f"Path to YOLO .pt weights (default: {_DEFAULT_MODEL.name}).",
    )
    parser.add_argument("-o", "--outdir", default="./output", help="Output directory.")
    parser.add_argument(
        "--conf", type=float, default=0.25, help="Min detection confidence (default: 0.25)."
    )
    args = parser.parse_args()

    from ultralytics import YOLO  # imported lazily so --help works without it installed

    model = YOLO(args.model)
    # Absolute path so ultralytics writes here, not under its own runs/ dir.
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    for image in args.images:
        results = model.predict(
            image, conf=args.conf, save=True, project=str(outdir), name=".", exist_ok=True
        )
        r = results[0]
        print(f"\n{Path(image).name}: {len(r.boxes)} detections")
        names = r.names
        for box in r.boxes:
            cls = names[int(box.cls)]
            conf = float(box.conf)
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            print(f"  {cls:<14} {conf:.2f}  bbox=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")
        print(f"  annotated image -> {outdir}/{Path(image).stem}.jpg")


if __name__ == "__main__":
    main()
