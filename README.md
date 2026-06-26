# Chess-Board-Detection-Classification

A three-stage computer vision pipeline that detects a chessboard from a photo, identifies the pieces, and recommends the best move using a chess engine.

## Pipeline Overview

| Stage | What it does | Output |
|-------|-------------|--------|
| Stage 1 | Detect the board and produce an 800x800 top-down warped image | `_warped.jpg`, `_grid.jpg`, `_debug.jpg` |
| Stage 2 | Run YOLO piece detection and generate a FEN string | `_pieces.jpg` |
| Stage 3 | Feed the FEN to Stockfish and get the best move | `_move.jpg` |

## Dependencies

### Stage 1 -- Board Detection (required)

```bash
pip install numpy opencv-python
```

### Stage 2 -- Piece Detection (optional)

```bash
pip install ultralytics
```

Also requires the YOLO model weights file: `chess-model-yolov8m.pt` (download separately).

### Stage 3 -- Engine Analysis (optional)

```bash
pip install chess
```

Also requires the Stockfish binary:

```bash
brew install stockfish   # macOS
sudo apt install stockfish   # Ubuntu/Debian
```

Or install from source: [stockfishchess.org](https://stockfishchess.org/download/)

### Install all at once

```bash
pip install -r requirements.txt
```

## Usage

### Stage 1 Only: Board Detection and Warping

Detects the board, warps it to a top-down 800x800 view, and draws the grid.

```bash
python -m so101_nexus_core.chess_vision Chess-Images/image.png -o ./output
```

Output files written to `./output/`:
- `image_warped.jpg` -- top-down warped board
- `image_grid.jpg` -- warped board with algebraic grid overlay
- `image_debug.jpg` -- original photo with detected corners projected back
- `image_debug_FAIL.jpg` -- written instead if detection fails

### Stages 1 and 2: Board Detection and Piece Classification

Detects the board and pieces, prints the FEN string.

```bash
python -m so101_nexus_core.chess_vision Chess-Images/image.png \
    --model chess-model-yolov8m.pt \
    -o ./output
```

Additional output:
- `image_pieces.jpg` -- original photo with YOLO bounding boxes and bottom-center markers

### Stages 1, 2 and 3: Full Pipeline with Move Recommendation

Detects the board, identifies all pieces, and asks Stockfish for the best move.

```bash
python -m so101_nexus_core.chess_vision Chess-Images/image.png \
    --model chess-model-yolov8m.pt \
    --stockfish ./Stockfish/src/stockfish \
    --turn w \
    -o ./output
```

Flags:
- `--turn {w,b}` -- whose turn it is (default: `w`)
- `--engine-time SECONDS` -- time Stockfish spends analysing (default: `1.0`)
- `--stockfish` -- path to the Stockfish binary; omit the path to use the one on `PATH`

Additional output:
- `image_move.jpg` -- warped grid with source and destination squares highlighted

### Run on multiple images at once

```bash
python -m so101_nexus_core.chess_vision Chess-Images/*.png --model chess-model-yolov8m.pt -o ./output
```

## Architecture

### Stage 1 -- Board Detection (`pipeline.py`)

Three independent detectors race and the best-scoring result wins:

- **SB detector** (`detect_sb.py`) -- OpenCV `findChessboardCornersSB`, reliable when 7x7 inner corners are visible.
- **Hough detector** (`detect_hough.py`) -- Canny edges and `HoughLinesP`, clusters lines into two perpendicular families and RANSAC-fits a homography.
- **Frame detector** (`detect_frame.py`) -- anchors on the outer boundary quad instead of interior corners; survives heavy piece occlusion.

Each coarse result is refined by `refine.py`, scored by `scoring.py` (checkerboard contrast on the warped image), and the single highest-scoring candidate is returned as a `Stage1Result` (`result.py`).

### Stage 2 -- Piece Detection (`pieces.py`)

Runs YOLO on the original unwarped photo (pieces are 3D objects and only look correct in their native camera angle). Projects each detection's bottom-center point through the Stage 1 homography into the 800x800 canvas, assigns it to an algebraic square, and builds a FEN string.

### Stage 3 -- Engine Analysis (`engine.py`)

Feeds the FEN to a Stockfish subprocess over UCI via `python-chess`. Returns the best move in both UCI and SAN notation and writes a highlighted move overlay onto the warped grid image.

## Project Structure

```
chess_vision/
├── pipeline.py          # Stage 1 entry point
├── pieces.py            # Stage 2 entry point
├── engine.py            # Stage 3 entry point
├── cli.py               # Command-line interface (__main__.py)
├── detect_sb.py         # SB corner detector
├── detect_hough.py      # Hough line detector
├── detect_frame.py      # Outer-boundary quad detector
├── refine.py            # Warp refinement
├── scoring.py           # Checkerboard scoring
├── geometry.py          # Pure geometry helpers
├── masking.py           # HSV board masking
├── overlay.py           # Debug drawing helpers
├── constants.py         # Shared tunables (WORK_LONG_EDGE, WARP_SIZE, SQ)
└── result.py            # Stage1Result dataclass
```

## License

Apache-2.0
