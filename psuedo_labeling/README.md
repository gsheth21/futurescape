# Pseudo Labeling

## What it does

Auto-labels game pieces in a directory of images using a two-stage pipeline:
1. A board detection model crops the board from each image
2. A piece detection model runs on the crop and outputs YOLO-format `.txt` label files

Supports two detection modes:
- **`single`** — for per-class models (e.g. yellow piece detector). All detections are written with a fixed `CLASS_ID`.
- **`multi`** — for the combined all-pieces model. Class IDs are taken directly from the model output.

This is used to generate pseudo-labels for unlabeled images, which can then be reviewed and corrected in Label Studio before being used for training.

---

## Pipeline Position

```
board_detection  ──┐
                   ├──► psuedo_labeling ──► (review in Label Studio) ──► piece_detection training
piece_detection ───┘
```

- **Requires**: trained board detector (`board_detector_v1.pt`) from `board_detection`, and a piece detector from `piece_detection` (any single-class detector, or `all_pieces_detector.pt` for multi mode)
- **Outputs**: YOLO `.txt` label files in `dataset/labels/`, one per image

---

## Setup

### 1. Install dependencies

```bash
pip install ultralytics opencv-python python-dotenv
```

### 2. Place model weights

- Board detector: obtain `board_detector_v1.pt` from `board_detection/saved_models/`
- **Single mode**: any per-class detector, e.g. `yellow_piece_detector.pt` from `piece_detection/src/yellow_piece_detection/saved_models/`
- **Multi mode**: `all_pieces_detector.pt` from `piece_detection/src/all_pieces_detection/saved_models/`

Model weights are not committed to the repo.

### 3. Configure `.env`

Edit `.env` in this directory and set all paths to absolute paths on your machine:

```
IMAGES_DIR=/absolute/path/to/psuedo_labeling/dataset/images
LABELS_DIR=/absolute/path/to/psuedo_labeling/dataset/labels
BOARD_MODEL_PATH=/absolute/path/to/board_detector_v1.pt
PIECE_MODEL_PATH=/absolute/path/to/piece_detector.pt
BOARD_CONF=0.3
PIECE_CONF=0.25

# Detection mode: 'single' (per-class model) or 'multi' (all-pieces model)
DETECTION_MODE=single

# Only used when DETECTION_MODE=single
CLASS_ID=0
```

| Variable | Description |
|---|---|
| `DETECTION_MODE` | `single` — fixed `CLASS_ID` for all detections; `multi` — class IDs from model output |
| `CLASS_ID` | YOLO class ID to write in label files (single mode only) |
| `PIECE_MODEL_PATH` | Path to any single-class detector **or** the all-pieces model |

### 4. Add images

Place images to auto-label in `dataset/images/`.

---

## How to Run

```bash
cd src
python semi_label.py
```

Output `.txt` files will appear in `dataset/labels/`, one per image, in YOLO format:
```
class_id cx cy w h
```

---

## Creating `classes.txt` and `notes.json`

Both files live in `dataset/` and tell the Label Studio converter what class names correspond to which class IDs. They must match the model you used.

### Single mode (per-class model)

One class per file. Set `name` to whatever the piece is called.

**`dataset/classes.txt`**
```
yellow_agriculture_piece
```

**`dataset/notes.json`**
```json
{
  "categories": [
    { "id": 0, "name": "Yellow Agriculture Piece" }
  ],
  "info": { "year": 2026, "version": "1.0", "contributor": "Label Studio" }
}
```

> If using a different single-class model (e.g. `red_piece_detector.pt` with `CLASS_ID=0`), replace the name with the appropriate piece name.

### Multi mode (all-pieces model)

All 7 classes in the exact order the model was trained with.

**`dataset/classes.txt`**
```
aquatic_life
population_resilient_more
military_installation
population_resilient_less
site_intervention
wildlife
yellow_agriculture_piece
```

**`dataset/notes.json`**
```json
{
  "categories": [
    { "id": 0, "name": "aquatic_life" },
    { "id": 1, "name": "population_resilient_more" },
    { "id": 2, "name": "military_installation" },
    { "id": 3, "name": "population_resilient_less" },
    { "id": 4, "name": "site_intervention" },
    { "id": 5, "name": "wildlife" },
    { "id": 6, "name": "yellow_agriculture_piece" }
  ],
  "info": { "year": 2026, "version": "1.0", "contributor": "Label Studio" }
}
```

> The class order in `classes.txt` must match the IDs in `notes.json` and the IDs the model outputs. Mismatches will cause wrong labels in Label Studio.

---

## Importing into Label Studio for Review

1. Convert labels to Label Studio JSON:
```bash
pip install label-studio-converter

label-studio-converter import yolo \
  -i /absolute/path/to/psuedo_labeling/dataset \
  --image-root-url "/data/local-files/?d=dataset/images" \
  -o output.json
```

2. Start Label Studio with local file serving:
```bash
docker run -it -p 8081:8080 \
  -v "${PWD}/mydata:/label-studio/data" \
  -v "/absolute/path/to/psuedo_labeling/dataset:/label-studio/files/dataset" \
  -e LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true \
  -e LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/label-studio/files \
  heartexlabs/label-studio:latest label-studio
```

**Note: Edit "/absolute/path/to/psuedo_labeling/dataset" according to your setup.**

3. In Label Studio: create a project → **Import** → upload `output.json`
