# Piece Detection

Detects all game pieces on the futurescape game board from a cropped map image. Includes an all-in-one 10-class detector and individual per-piece detectors, plus utilities for dataset creation and label visualization.

## Pipeline Position

Receives cropped map images from the **map detection** step. Outputs bounding boxes (class + coordinates) for each piece, consumed by `final_pipeline/`.

```
map_detection  ──►  piece_detection  ──►  final_pipeline
```

## Repository Dataset

Datasets are **not** stored in this repo (too large). Download from Google Drive:

> **Dataset link:** _[placeholder — add Google Drive link here]_

Expected layout after download:

```
datasets/
├── year/                         ← year label images and labels
│   ├── images/
│   └── labels/
└── pieces_dataset/
    ├── all_pieces/               ← merged dataset (output of dataset_creator.py)
    ├── aquatic_life/
    ├── black_pieces/
    ├── military_installation/
    ├── red_pieces/
    ├── site_intervention/
    ├── wildlife/
    └── yellow_pieces/
```

Each label subdirectory contains `images/`, `labels/`, and `notes.json` directly.

## Setup

### 1. Install dependencies

```bash
pip install ultralytics python-dotenv opencv-python pyyaml torch
```

### 2. Configure paths

Copy `.env` and fill in absolute paths for your machine:

```bash
cp .env .env  # already present — just edit it
```

Open `.env` and set:

| Variable | Description |
|---|---|
| `DATASET_DIR` | Root directory containing all datasets |
| `PIECES_DATASET_DIR` | Expands to `$DATASET_DIR/pieces_dataset` — root of all piece label folders |
| `ALL_PIECES_DATASET_DIR` | Expands to `$PIECES_DATASET_DIR/all_pieces` — merged YOLO dataset |
| `AQUATIC_LIFE_LABELS_DIR` | Expands to `$PIECES_DATASET_DIR/aquatic_life` |
| `BLACK_LABELS_DIR` | Expands to `$PIECES_DATASET_DIR/black_pieces` |
| `MILITARY_LABELS_DIR` | Expands to `$PIECES_DATASET_DIR/military_installation` |
| `RED_LABELS_DIR` | Expands to `$PIECES_DATASET_DIR/red_pieces` |
| `SITE_INTERVENTION_LABELS_DIR` | Expands to `$PIECES_DATASET_DIR/site_intervention` |
| `WILDLIFE_LABELS_DIR` | Expands to `$PIECES_DATASET_DIR/wildlife` |
| `YEAR_LABELS_DIR` | Expands to `$DATASET_DIR/year` (lives at root, not under `pieces_dataset`) |
| `YELLOW_LABELS_DIR` | Expands to `$PIECES_DATASET_DIR/yellow_pieces` |
| `BOARD_DETECTOR_MODEL` | Absolute path to `board_detector_v1.pt` (from `superpoint` module) |

### 3. Model weights

Place trained detector weights inside the corresponding `saved_models/` directory for each sub-pipeline. Weight files (`*.pt`) are not committed to version control.

| Sub-pipeline | Expected weight filename |
|---|---|
| `all_pieces_detection/` | `saved_models/all_pieces_detector.pt` |
| `aquatic_life_piece_detection/` | `saved_models/aquatic_life_piece_detector.pt` |
| `black_piece_detection/` | `saved_models/black_piece_detector.pt` |
| `military_installation_detection/` | `saved_models/military_installation_detector.pt` |
| `red_piece_detection/` | `saved_models/red_piece_detector.pt` |
| `site_intervention_detection/` | `saved_models/site_intervention_detector.pt` |
| `wildlife_piece_detection/` | `saved_models/wildlife_piece_detector.pt` |
| `year_detection/` | `saved_models/year_detector.pt` |
| `yellow_piece_detection/` | `saved_models/yellow_piece_detector.pt` |

## Source Layout

```
src/
├── all_pieces_detection/          ← single 10-class detector (recommended for pipeline)
│   └── detection.ipynb
├── aquatic_life_piece_detection/  ← individual aquatic life piece detector
│   └── detection.ipynb
├── black_piece_detection/         ← individual black piece detector
│   └── detection.ipynb
├── military_installation_detection/ ← individual military installation detector
│   └── detection.ipynb
├── red_piece_detection/           ← individual red piece detector
│   └── detection.ipynb
├── site_intervention_detection/   ← individual site intervention detector
│   └── detection.ipynb
├── wildlife_piece_detection/      ← individual wildlife piece detector
│   └── detection.ipynb
├── year_detection/                ← year token detector
│   └── detection.ipynb
├── yellow_piece_detection/        ← individual yellow piece detector
│   └── detection.ipynb
├── data_creation/
│   └── dataset_creator.py         ← merges per-class label dirs into one YOLO dataset
└── data_visualization/
    └── labels_viz.py              ← visualize YOLO labels on images
```

## How to Run

### All-pieces detector (recommended)

Open and run `src/all_pieces_detection/detection.ipynb` cell by cell:

1. **Step 1** — loads dataset paths from `.env`
2. **Step 2** — crops images to board bounding box using the board detector model
3. **Step 3** — train/val/test split
4. **Step 4** — creates `data.yaml` for YOLO
5. **Step 5** — trains YOLOv8 model
6. **Step 6** — validates and exports the best checkpoint to `saved_models/`

### Per-piece detectors

Each sub-pipeline in `src/<piece_name>_detection/detection.ipynb` follows the same step structure as above but trains a single-class binary detector.

### Dataset Creator

Merges all per-class label subdirectories into a single YOLO dataset under `pieces_dataset/all_pieces/`.

```bash
python src/data_creation/dataset_creator.py
```

Reads `PIECES_DATASET_DIR` from `.env`. Outputs merged `images/`, `labels/`, `classes.txt`, and `notes.json`.

### Label Visualization

Visualize YOLO bounding box labels on images. Expected dataset layout:

```
dataset_dir/
    classes.txt
    images/  *.jpg | *.png
    labels/  *.txt
```

**Single image (interactive window):**

```bash
python src/data_visualization/labels_viz.py show \
    <image_path> <label_path> <classes_txt_path>
```

**Batch — save annotated images to a directory:**

```bash
python src/data_visualization/labels_viz.py batch \
    <dataset_dir> <output_dir>
```

### Batch Mode (Save Annotated Images)

Runs through every image in a dataset and saves annotated outputs.

```bash
python data_visualization/labels_viz.py batch <dataset_dir> <output_dir>
```

## Notes

- The two-stage approach (map first, then pieces) reduces false detections outside the map.
- Prediction outputs include bounding boxes for all detected pieces; in two-stage mode, convert crop coordinates back to original image coordinates using crop offsets.
