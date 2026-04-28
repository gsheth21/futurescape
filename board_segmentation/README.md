# Board Segmentation Module

## What It Does

Trains a YOLOv8 instance segmentation model to produce **pixel-level polygon masks** of the
board in each image. Unlike bounding boxes, the output mask conforms to the actual board
shape, enabling precise corner extraction and perspective correction downstream.

Two training notebooks are provided:
- `src/yolo_seg.ipynb` — primary training notebook with diagnostics and crash recovery helpers
- `src/polyseg_datasug.py` — polygon-aware data augmentation (geometric + photometric transforms)

---

## Pipeline Position

```
Input Photo
    │
    ▼
[board_detection]  ──→  bounding box crop
    │
    ▼
[board_segmentation]  ──→  polygon mask  ──→  corner keypoints (board_keypoint_prediction)
    │
    ▼
[homography]  ──→  rectified 2D board view
```

The segmentation mask feeds directly into the homography step to compute the perspective
transform that flattens the board image into a canonical overhead view.

---

## How to Run

### 1. Data Augmentation

```bash
cd board_segmentation
python src/polyseg_datasug.py
```

Reads raw images and YOLO-format polygon labels from `AUG_DATASET_DIR`, writes augmented
images and labels to `<AUG_DATASET_DIR>/images_aug/` and `<AUG_DATASET_DIR>/labels_aug/`.

Tunable constants (edit in `src/polyseg_datasug.py` directly):

| Constant  | Default | Meaning                            |
|-----------|---------|------------------------------------|
| `N_AUGS`  | `10`    | Augmented copies per source image  |

### 2. Training

Open `src/yolo_seg.ipynb` in Jupyter / VS Code and run cells top-to-bottom.

Key hyperparameters (hardcoded in the training-config cell):

| Constant         | Default | Meaning                   |
|------------------|---------|---------------------------|
| `EPOCHS`         | `40`    | Training epochs           |
| `IMAGE_SIZE`     | `320`   | YOLO input resolution     |
| `BATCH_SIZE`     | `6`     | Samples per gradient step |
| `LEARNING_RATE`  | `0.01`  | SGD initial LR            |
| `RUN_NAME`       | `"board_segmentation_80"` | Experiment subfolder |

### 3. Inference / Evaluation

Both notebooks include inference and evaluation cells that run on the test split.
After training, the best model weights are saved to:

```
runs/<RUN_NAME>/weights/best.pt
```

---

## Setup

### Environment

```bash
conda create -n board_seg python=3.9 -y
conda activate board_seg
pip install -r requirements.txt
```

### `.env` Configuration

Copy `.env` and adjust paths for your machine. The file is **not committed** to version
control — each developer maintains their own copy.

| Variable           | Default value                    | Purpose                                   |
|--------------------|----------------------------------|-------------------------------------------|
| `SEG_BASE_DIR`     | `..`                             | Path to `board_segmentation/` root        |
| `AUG_DATASET_DIR`  | `../board_segmentation_dataset`  | Dataset root for augmentation script      |
| `PRETRAINED_MODEL` | `yolov8m-seg.pt`                 | YOLO seg weights to fine-tune from        |

### Pre-trained Weights

Download `yolov8m-seg.pt` from Ultralytics:

```bash
# From the board_segmentation/ directory:
python -c "from ultralytics import YOLO; YOLO('yolov8m-seg.pt')"
```

Or download directly from
https://github.com/ultralytics/assets/releases

### Directory Structure

```
board_segmentation/
├── .env                          ← machine-specific config (not committed)
├── README.md
├── requirements.txt
├── yolov8m-seg.pt                ← pre-trained weights (download separately)
├── src/
│   ├── yolo_seg.ipynb            ← training notebook
│   └── polyseg_datasug.py        ← polygon data augmentation
├── board_segmentation_dataset/
│   ├── images/                   ← original images
│   ├── labels/                   ← YOLO polygon labels (.txt)
│   ├── images_aug/               ← output of polyseg_datasug.py
│   ├── labels_aug/               ← output of polyseg_datasug.py
│   └── classes.txt
├── yolo_segmentation_dataset/
│   ├── train/
│   ├── val/
│   └── test/
└── runs/                         ← training outputs (weights, plots, metrics)
```
