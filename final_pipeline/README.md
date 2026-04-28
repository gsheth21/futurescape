# final_pipeline — End-to-End Board Game State Detection

## What It Does

Takes a raw photo of the board game and outputs a `hexid → piece` occupancy map. Ties together every upstream module into a single runnable script.

---

## Where It Fits in the Pipeline

```
board_detection        →  detects and crops the board region
superpoint             →  matches crop to ideal template, projects hex grid
piece_detection        →  detects red pieces on the cropped board
final_pipeline         →  combines all of the above → hex-piece map (this module)
```

**Inputs:** raw `.jpg` / `.jpeg` / `.png` photo(s) of the board  
**Outputs per image:**
- `<name>_crop.png` — cropped board region
- `<name>_hex_piece_map.json` — `{ "hex_001": 0, "hex_002": 1, ... }` (`1` = occupied, `0` = empty)
- `<name>_hex_piece_viz.png` — annotated visualization (piece boxes + hex centers)

---

## How to Run

```bash
cd final_pipeline/src

# Single image
python pipeline.py --image path/to/photo.jpg --output_dir path/to/output/

# Directory of images
python pipeline.py --image_dir path/to/photos/ --output_dir path/to/output/
```

Supported input formats: `.jpg`, `.jpeg`, `.png`

All model paths and tunable parameters are read from `final_pipeline/.env` — edit that file before running (see Setup below).

---

## Setup

### 1. Install dependencies

```bash
pip install ultralytics opencv-python numpy torch python-dotenv
```

### 2. Clone SuperPoint

The pipeline uses the MagicLeap SuperPoint implementation. Clone it into the `superpoint/` directory:

```bash
git clone https://github.com/magicleap/SuperPointPretrainedNetwork.git \
    superpoint/SuperPointPretrainedNetwork
```

### 3. Place model weights

The following files are **not** tracked in git. Download / copy them and update the paths in `.env`:

| File | Description |
|---|---|
| `board_detector_v1.pt` | YOLO model from `board_detection/saved_models/` |
| `red_piece_detector.pt` | YOLO model from `piece_detection/saved_models/` |
| `superpoint_v1.pth` | Pretrained SuperPoint weights (from the cloned repo above) |
| `cropped_ideal_image.png` | Ideal template image (from `superpoint/raw_dataset/`) |
| `cropped_ideal_image_gt.npy` | Template keypoints `.npy` |
| `cropped_ideal_image_hex_centers.json` | Hex center coordinates on the ideal template |

### 4. Configure `.env`

Edit `final_pipeline/.env` with the absolute paths for your machine:

```
SUPERPOINT_SRC_DIR=/absolute/path/to/superpoint/src
BOARD_YOLO_WEIGHTS=/absolute/path/to/board_detector_v1.pt
PIECE_YOLO_WEIGHTS=/absolute/path/to/red_piece_detector.pt
SP_WEIGHTS=/absolute/path/to/superpoint_v1.pth
TEMPLATE_DIR=/absolute/path/to/cropped_ideal_templates/
HEX_CENTERS_PATH=/absolute/path/to/cropped_ideal_image_hex_centers.json
BOARD_CONF=0.5
PIECE_CONF=0.5
BOARD_PAD=0.05
```

---

## Pipeline Steps

```
Raw Image
    │
    ▼
[1] Map Detection (YOLO)
    └─ Crops the board region from the full image (+5% padding)
    │
    ▼
[2] SuperPoint Feature Matching
    └─ Resizes crop to match template preprocessing scale (max 960px long side)
    └─ Extracts keypoints & descriptors, matches to template, computes homography
    └─ Projects template hex centers into crop space, scales back to original resolution
    │
    ▼
[3] Piece Detection (YOLO)
    └─ Runs on the original (unresized) crop
    └─ Returns bounding boxes + confidence scores
    │
    ▼
[4] Piece → Hex Assignment
    └─ Each piece center is matched to the nearest hex center within 2 × adaptive radius
    └─ Radius = 60% of median nearest-neighbour distance between hex centers
    │
    ▼
[5] Outputs saved to --output_dir
```
