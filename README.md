# FutureScape — Board Game State Detection

Computer vision pipeline for the [FutureScape](https://iexcel.ncsu.edu/) board game, developed at the iEXCEL Lab at NC State University. Given a raw photograph taken during gameplay, the system outputs a structured `hex_id → piece` occupancy map — no manual observation required.

## Problem

Human observers currently record piece placements by hand after each round — slow, error-prone, and impossible to scale to remote sessions. This pipeline automates that process from a single photo.

## Production Pipeline

```
Raw Photo
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. Map Detection  (map_detection/)                          │
│    YOLOv8n fine-tuned → crops the hexagonal map region      │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. SuperPoint Feature Matching  (superpoint/)               │
│    Matches crop to ideal template via descriptor matching   │
│    + RANSAC → projects all hex centers into image space     │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Piece Detection  (piece_detection/)                      │
│    YOLOv8 fine-tuned → bounding boxes for all piece types   │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Piece → Hex Assignment  (final_pipeline/)                │
│    Nearest-neighbor matching → hex_piece_map.json           │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
Output: { "hex_001": 1, "hex_002": 0, ... }
```

## Experimental Modules

The following modules were built to explore alternative approaches and are included as standalone training pipelines:

| Module | Approach | Notes |
|---|---|---|
| `board_detection/` | YOLOv8n bounding-box detector for the full board | Prototype; led to `map_detection` |
| `board_segmentation/` | YOLOv8 instance segmentation for precise board mask + corners | Exploratory |
| `board_keypoint_prediction/` | YOLOv8-pose 9-keypoint predictor for homography computation | Prior to SuperPoint |

`pseudo_labeling/` is a data utility that auto-labels new images using trained detectors to speed up annotation in Label Studio.

## Repository Structure

```
FutureScape/
├── README.md
├── requirements.txt
│
├── map_detection/              # Step 1 — detect and crop the map region
│   ├── README.md
│   ├── src/
│   └── .env
│
├── superpoint/                 # Step 2 — feature matching + hex center projection
│   ├── README.md
│   ├── src/
│   └── .env
│
├── piece_detection/            # Step 3 — detect game pieces on the map crop
│   ├── README.md
│   ├── src/
│   └── .env
│
├── final_pipeline/             # Step 4 — end-to-end runner → hex_piece_map.json
│   ├── README.md
│   ├── src/
│   └── .env
│
├── pseudo_labeling/            # Utility — auto-label images for training
│   ├── README.md
│   ├── src/
│   └── .env
│
├── board_detection/            # Experimental — YOLOv8 board bounding box
│   ├── README.md
│   ├── src/
│   └── .env
│
├── board_segmentation/         # Experimental — YOLOv8 instance segmentation
│   ├── README.md
│   ├── src/
│   └── .env
│
└── board_keypoint_prediction/  # Experimental — YOLOv8-pose keypoint detection
    ├── README.md
    ├── src/
    └── .env
```

Datasets, model weights, training outputs, and raw images are **not committed**. See `.gitignore` and each module's README for download/placement instructions.

## Getting Started

### Prerequisites

- Python 3.9+
- NVIDIA GPU with CUDA 12+ (CPU fallback supported)
- Conda (recommended)

### Install

```bash
git clone https://github.com/gpsheth_ncstate/FutureScape.git
cd FutureScape

conda create -n futurescape python=3.9
conda activate futurescape

pip install -r requirements.txt
```

### Clone SuperPoint weights

```bash
git clone https://github.com/magicleap/SuperPointPretrainedNetwork.git \
    superpoint/SuperPointPretrainedNetwork
```

### Run the pipeline on an image

```bash
cd final_pipeline/src
python pipeline.py --image path/to/photo.jpg --output_dir path/to/output/
```

See [final_pipeline/README.md](final_pipeline/README.md) for full setup: `.env` configuration and required model weight locations.

## Module Summary

| Module | What it does | Feeds into | Docs |
|---|---|---|---|
| `map_detection` | YOLOv8n fine-tuned to detect and crop the hexagonal map region | `superpoint`, `final_pipeline` | [README](map_detection/README.md) |
| `superpoint` | SuperPoint feature matching against ideal template; projects all hex centers onto the image | `final_pipeline` | [README](superpoint/README.md) |
| `piece_detection` | YOLOv8 multi-class detector for all 10 piece types | `final_pipeline` | [README](piece_detection/README.md) |
| `final_pipeline` | End-to-end runner: map crop → hex projection → piece detection → JSON output | — | [README](final_pipeline/README.md) |
| `pseudo_labeling` | Two-stage auto-labeler: board detector crops, piece detector labels → YOLO `.txt` files | `piece_detection` training | [README](psuedo_labeling/README.md) |
| `board_detection` | YOLOv8n bounding-box detector for the full board *(experimental)* | — | [README](board_detection/README.md) |
| `board_segmentation` | YOLOv8 instance segmentation for precise board mask *(experimental)* | — | [README](board_segmentation/README.md) |
| `board_keypoint_prediction` | YOLOv8-pose 9-keypoint predictor for perspective mapping *(experimental)* | — | [README](board_keypoint_prediction/README.md) |

## Dataset

~350 real-world images were captured across 5 gameplay sessions (Fall 2025) with undergraduate students in ENV 101 and HON 293 at NC State University. Datasets are not stored in this repo due to size — see each module's README for expected directory layout.

## Contributors

- Dee Kandel
- Ashwin Mahesh
- Petr Shvarev
- Justin Huang
- Gaurav Sheth
- Dr. Aditi Mallavarapu (iEXCEL Lab, NC State University)
