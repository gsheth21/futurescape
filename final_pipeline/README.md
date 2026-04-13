# Final Pipeline — Board Game State Detection

End-to-end pipeline that takes a raw photo of the board game and outputs a `hexid → piece` occupancy map.

---

## Pipeline Overview

```
Raw Image
    │
    ▼
[1] Board Detection (YOLO)
    └─ Crops the board region from the full image (+5% padding)
    │
    ▼
[2] SuperPoint Feature Matching
    └─ Resizes crop to match template preprocessing scale (max 960px long side)
    └─ Extracts keypoints & descriptors from crop and ideal template
    └─ Matches features, computes homography
    └─ Projects template hex centers into cropped image space
    └─ Scales projected coordinates back to original crop space
    │
    ▼
[3] Red Piece Detection (YOLO)
    └─ Runs directly on the original (unresized) crop
    └─ Returns bounding boxes + confidence scores
    │
    ▼
[4] Piece → Hex Assignment
    └─ Computes center of each piece bounding box
    └─ For each piece: finds the nearest hex center within threshold (2 × radius)
    └─ Each piece assigned to exactly 1 hex (nearest); multiple pieces can share a hex
    └─ Threshold radius = 60% of median nearest-neighbour distance between hex centers
    │
    ▼
[5] Outputs
    └─ <name>_crop.png          — cropped board image
    └─ <name>_hex_piece_map.json — { "hex_001": 0, "hex_002": 1, ... }
    └─ <name>_hex_piece_viz.png  — annotated visualization
```

---

## Output Format

### `<name>_hex_piece_map.json`
```json
{
  "hex_001": 0,
  "hex_002": 1,
  "hex_003": 0,
  ...
}
```
- `1` — hex cell is occupied by a red piece
- `0` — hex cell is empty

### `<name>_hex_piece_viz.png`
Annotated image showing:
- **Blue boxes** — detected piece bounding boxes with confidence score
- **Green dots** — occupied hex centers
- **Grey dots** — empty hex centers
- Hex IDs labelled next to each dot

---

## Usage

```bash
# Single image
python pipeline.py --image path/to/photo.jpg --output_dir path/to/output/

# Directory of images
python pipeline.py --image_dir path/to/photos/ --output_dir path/to/output/
```

Supported input formats: `.jpg`, `.jpeg`, `.png`

---

## Hardcoded Paths (edit in `pipeline.py` if needed)

| Variable | Description |
|---|---|
| `BOARD_YOLO_WEIGHTS` | YOLO model for board/map detection |
| `PIECE_YOLO_WEIGHTS` | YOLO model for red piece detection |
| `SP_WEIGHTS` | Pretrained SuperPoint weights |
| `TEMPLATE_DIR` | Directory containing the ideal template image and `.npy` files |
| `HEX_CENTERS_PATH` | JSON file with hex center coordinates on the ideal template |

---

## Key Design Decisions

### Coordinate Space Alignment
SuperPoint requires the test image to be preprocessed identically to the ideal template (resized so long side ≤ 960px). The pipeline resizes the crop for SuperPoint, then **scales the projected hex centers back** to original crop space so they align with piece detections, which run on the unresized crop.

### Piece–Hex Assignment
- **One piece → one hex**: each piece is assigned to its single nearest hex center
- **One hex → many pieces**: a hex is marked occupied if any piece maps to it  
- **Threshold**: a piece is only eligible for assignment if its center is within `2 × radius` of a hex center, where `radius` adapts dynamically to the projected hex spacing (handles perspective distortion at different zoom levels)

### Model Loading
All models (board YOLO, piece YOLO, SuperPoint) and the template are loaded **once** at startup and reused across all images in batch mode.

---

## Dependencies

```
ultralytics
opencv-python
numpy
torch
```

SuperPoint also requires the `SuperPointPretrainedNetwork` repo cloned at `../SuperPointPretrainedNetwork/` relative to `../superpoint/src/`.
