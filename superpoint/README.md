# SuperPoint Board Alignment

## What It Does

Uses MagicLeap's pretrained SuperPoint network to align a board game image against a canonical ideal template via feature matching and RANSAC homography estimation. Once aligned, canonical keypoints (and optionally hex cell centers) defined on the template are projected into the coordinate space of each new image.

Two pipelines are provided:

- **no_crop** — runs SuperPoint directly on full-size images
- **with_crop** — first detects and crops the board region with YOLO, then runs SuperPoint on the crop

## Where It Fits in the Pipeline

```
map_detection/          ← trains the YOLO board detector used by with_crop/
      ↓
superpoint/             ← this module
      ↓
final_pipeline/         ← consumes *_pred.npy keypoints;
                           compute_homography.py and project_hex.py live there
```

## Directory Structure

```
superpoint/
├── .env                        ← machine-specific paths
├── README.md
└── src/
    ├── shared/                 ← core utilities shared by both pipelines
    │   ├── superpoint.py       ← model loading + inference
    │   ├── wrapper.py          ← class-based matcher with rich visualizations
    │   ├── matcher.py          ← mutual nearest-neighbor matching + RANSAC homography
    │   ├── predictor.py        ← project canonical keypoints through homography H
    │   ├── preprocessing.py    ← image conversion, resize, normalize, tensor loading
    │   ├── visualization.py    ← draw keypoints, matches, PCK threshold rings
    │   ├── keypoint_helper.py  ← interactive click-to-annotate GT keypoints → *_gt.npy
    │   └── extract.py          ← YOLO board detection + padded crop
    │
    ├── no_crop/                ← full-image pipeline
    │   ├── run_pipeline.py     ← end-to-end: load → detect → match → project → save
    │   └── evaluator.py        ← MED + PCK metrics against manually clicked GT
    │
    ├── with_crop/              ← YOLO-crop-first pipeline
    │   ├── run_pipeline_cropped.py  ← YOLO crop → SuperPoint → project → save
    │   └── cropped_eval.py          ← metrics with automatic crop-offset GT adjustment
    │
    └── canonical_template/
        └── hex_marker.py       ← interactive tool to mark hex centers on template → *_hex_centers.json
```

## How the Pipeline Works

### Full-image pipeline (`no_crop/`)

1. Load grayscale float tensors for the ideal template and all test images.
2. Run SuperPoint on both to get keypoints and 256-D descriptors.
3. Match descriptors via mutual nearest-neighbor filtering.
4. Estimate homography with RANSAC from matched pairs.
5. Project canonical keypoints (clicked on template via `keypoint_helper.py`) through the homography.
6. Save predictions as `*_pred.npy` and optional visualizations.

### Cropped pipeline (`with_crop/`)

Same as above, but step 0 is: detect the board bounding box with YOLO, crop + pad the region, and run SuperPoint only on that crop. The crop bbox is saved as `*_bbox.npy` so `cropped_eval.py` can convert full-image GT coords into crop space before computing metrics.

## Setup

### 1. Clone SuperPoint pretrained network

```bash
git clone https://github.com/magicleap/SuperPointPretrainedNetwork.git
```

Place the cloned folder at `superpoint/SuperPointPretrainedNetwork/` (or update `SP_WEIGHTS` in `.env`).

### 2. Model weights

| Weight | Source |
|---|---|
| `superpoint_v1.pth` | inside `SuperPointPretrainedNetwork/pretrained/` after clone |
| `board_detector_v1.pt` | trained in `map_detection/` module |

### 3. Configure `.env`

Copy `.env` and fill in your local paths:

```
SP_WEIGHTS=<path to superpoint_v1.pth>
YOLO_BOARD_WEIGHTS=<path to board_detector_v1.pt>

TEMPLATE_DIR=<path to ideal template folder>
TEST_IMAGE_DIR=<path to test images folder>
RESULTS_DIR=<where to write outputs>
CANONICAL_KPS_PATH=<path to ideal_image_gt.npy>
GT_DIR=<path to folder with *_gt.npy ground truth files>
PRED_DIR=<path to folder where *_pred.npy are written>

CROPPED_TEMPLATE_DIR=<path to cropped ideal template folder>
CROPPED_TEST_IMAGE_DIR=<path to full-size test images for cropped pipeline>
CROPPED_RESULTS_DIR=<where cropped pipeline writes outputs>
CROPPED_CANONICAL_KPS_PATH=<path to cropped_ideal_image_gt.npy>
CROPPED_GT_DIR=<path to GT folder for cropped eval>
CROPPED_PRED_DIR=<path to pred folder for cropped eval>
```

### 4. Prepare template data

1. Pick an ideal board image and preprocess it (PNG, grayscale, resized to max 960px long side) using `preprocessing.py`.
2. Run `keypoint_helper.py` to click canonical keypoints on the template → saves `*_gt.npy`.
3. Run `hex_marker.py` (in `canonical_template/`) to click hex cell centers → saves `*_hex_centers.json`.

## How to Run

### Annotate template keypoints

```bash
cd src/shared
python keypoint_helper.py --image <path_to_ideal_template.png>
```

### Run full-image pipeline

```bash
cd src/no_crop
python run_pipeline.py
```

### Run cropped pipeline

```bash
cd src/with_crop
python run_pipeline_cropped.py
```

### Evaluate

```bash
# Full-image pipeline
cd src/no_crop
python evaluator.py

# Cropped pipeline
cd src/with_crop
python cropped_eval.py
```

### Annotate hex centers (for projection downstream)

```bash
cd src/canonical_template
python hex_marker.py --image <path_to_cropped_ideal_template.png>
```
