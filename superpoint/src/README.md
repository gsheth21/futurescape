# SuperPoint Board Alignment and Hex Projection (src)

This directory contains a full feature-matching and homography pipeline for board images using SuperPoint keypoints/descriptors, optional YOLO-based board cropping, and downstream projection of canonical template coordinates (including hex centers) into each test image.

The code is organized to support:

- keypoint detection and matching with SuperPoint
- robust homography estimation with RANSAC
- canonical keypoint projection into new images
- optional YOLO board detection + crop-first inference
- quantitative evaluation against manually clicked keypoints
- homography and hex-center visualization outputs

## Directory Overview

Top-level scripts in this folder:

- `superpoint.py`: SuperPoint model loading and inference helpers
- `matcher.py`: descriptor matching + homography estimation
- `predictor.py`: homography-based projection of canonical points
- `preprocessing.py`: image conversion, resize, normalization, tensor loading
- `visualization.py`: keypoint/match/PCK visualization helpers
- `extract.py`: YOLO board detection and crop utilities
- `run_pipeline.py`: baseline pipeline on full images
- `run_pipeline_cropped.py`: crop-first pipeline (YOLO + SuperPoint)
- `evaluator.py`: metric report for full-image pipeline outputs
- `cropped_eval.py`: metric report for cropped pipeline outputs (with bbox adjustment)
- `keypoint_helper.py`: interactive keypoint annotation tool (`*_gt.npy`)
- `wrapper.py`: class-based SuperPoint matcher with rich visual diagnostics

Subfolders:

- `canonical_template/`
	- `hex_marker.py`: interactive hex-center annotation tool
- `homography/`
	- `compute_homography.py`: visualize correspondences + warp overlays
- `projection/`
	- `project_hex.py`: project template hex centers to each test image

Model artifacts:

- `saved_models/board_detector_v1.pt`
- `yolo11n.pt`, `yolov8n.pt` (local detector checkpoints)
- SuperPoint weights expected externally by default:
	- `../SuperPointPretrainedNetwork/pretrained/superpoint_v1.pth`

## How the Pipeline Works

At a high level:

1. Build grayscale float tensors for template and test images.
2. Run SuperPoint to obtain keypoints and descriptors.
3. Match descriptors via mutual nearest-neighbor filtering.
4. Estimate homography with RANSAC from matched keypoints.
5. Project canonical template keypoints through the homography.
6. Save predictions (`*_pred.npy`) and optional visualizations.

For the cropped pipeline:

1. Detect board bounding box with YOLO.
2. Crop board region (+padding), then run SuperPoint matching on crop.
3. Save crop-space predictions and crop bbox (`*_bbox.npy`) for evaluation.

## Environment and Dependencies

Minimum runtime stack used by these scripts:

- Python 3.9+
- PyTorch
- OpenCV (`opencv-python`)
- NumPy
- Pillow
- Matplotlib
- Ultralytics (`ultralytics`)

Install example:

```bash
pip install numpy opencv-python pillow matplotlib torch ultralytics
```

If your environment already has project-wide dependencies, use those instead.

## Data and File Conventions

The scripts assume consistent naming:

- image: `<name>.png`
- ground truth keypoints: `<name>_gt.npy`
- predicted keypoints: `<name>_pred.npy`
- crop image (cropped pipeline): `<name>_crop.png`
- crop bounding box (cropped pipeline): `<name>_bbox.npy`

Keypoint formats:

- manual/ground-truth canonical keypoints: `(N, 2)` as `(x, y)`
- SuperPoint keypoints from model: `(3, N)` as `(x, y, confidence)`
- SuperPoint descriptors: `(256, N)`

## Quick Start

### 1) Prepare canonical template keypoints

Click canonical keypoints once on your template image:

```bash
python keypoint_helper.py --image ../raw_dataset/preprocessed/ideal_templates/ideal_image.png
```

This saves `ideal_image_gt.npy` next to the template image.

### 2) (Optional) Normalize images to `.npy`

```bash
python preprocessing.py
```

The helper functions in `preprocessing.py` can also convert to grayscale, resize, and inspect sizes.

### 3) Run baseline (full-image) prediction pipeline

```bash
python run_pipeline.py
```

Default behavior:

- loads canonical keypoints from `CANONICAL_KPS_PATH`
- loads one template from `TEMPLATE_DIR`
- processes each image in `TEST_IMAGE_DIR`
- saves projected keypoints to `RESULTS_DIR` as `*_pred.npy`
- writes match and prediction visualizations to `RESULTS_DIR`

### 4) Evaluate baseline predictions

First click GT for test images:

```bash
python keypoint_helper.py --all ../raw_dataset/preprocessed/new_test_images
```

Then run evaluation:

```bash
python evaluator.py
```

Reported metrics include:

- mean and median Euclidean distance
- PCK at multiple pixel thresholds
- per-keypoint mean error across dataset

## Cropped Pipeline (YOLO + SuperPoint)

Use this when board detection and cropping improves matching robustness.

### 1) Run crop-first inference

```bash
python run_pipeline_cropped.py
```

What it does:

- detects board bbox with YOLO (`extract.py`)
- crops board region with padding
- runs SuperPoint matching on cropped image
- saves:
	- `*_pred.npy` (predicted keypoints in crop coordinate space)
	- `*_bbox.npy` (crop bbox in full-image coordinates)
	- visual outputs (`*_crop.png`, `*_matches.png`, `*_pred.png`)

### 2) Evaluate cropped predictions

```bash
python cropped_eval.py
```

`cropped_eval.py` automatically adjusts GT from full-image coordinates to crop coordinates using each saved `*_bbox.npy`.

## Homography and Hex Projection Utilities

### Homography visualization

Single image:

```bash
python homography/compute_homography.py \
	--template_dir ../raw_dataset/preprocessed/cropped_ideal_templates \
	--test_image ../raw_dataset/preprocessed/cropped_test_images_results/example_crop.png \
	--output_dir homography/output/homography_viz
```

Directory mode:

```bash
python homography/compute_homography.py \
	--template_dir ../raw_dataset/preprocessed/cropped_ideal_templates \
	--test_dir ../raw_dataset/preprocessed/cropped_test_images_results \
	--output_dir homography/output/homography_viz
```

Outputs:

- `*_matches.png` (inliers/outliers visualization)
- `*_warp.png` (template warp overlay)

### Mark canonical hex centers

```bash
python canonical_template/hex_marker.py --image ../raw_dataset/preprocessed/cropped_ideal_templates/cropped_ideal_image.png
```

Creates `cropped_ideal_image_hex_centers.json`.

### Project template hex centers to test images

Single image:

```bash
python projection/project_hex.py \
	--template_dir ../raw_dataset/preprocessed/cropped_ideal_templates \
	--test_image ../raw_dataset/preprocessed/cropped_test_images_results/example_crop.png \
	--output_dir projection/output/hex_projections
```

Directory mode:

```bash
python projection/project_hex.py \
	--template_dir ../raw_dataset/preprocessed/cropped_ideal_templates \
	--test_dir ../raw_dataset/preprocessed/cropped_test_images_results \
	--output_dir projection/output/hex_projections
```

Outputs:

- `*_hex.json`: projected hex center coordinates
- `*_hex_viz.png`: projected centers overlaid on the test crop

## Core Module API Summary

### `superpoint.py`

- `load_model(weights_path, cuda=False)`
	- loads `SuperPointFrontend`
- `detect(fe, tensor)`
	- returns `(pts, desc, scores)`

### `matcher.py`

- `match(desc1, desc2, nn_thresh=0.7)`
	- mutual nearest-neighbor descriptor matching
- `compute_homography(pts_template, pts_test, matches)`
	- RANSAC homography from matched points

### `predictor.py`

- `project_keypoints(canonical_kps, H)`
	- projects `(N, 2)` points from template to image space

### `extract.py`

- `detect_board(image_path, weights, conf_thresh=0.5)`
- `add_padding(bbox, image_shape, padding=0.05)`
- `crop_board(image_path, weights, padding=0.05)`

### `preprocessing.py`

- `convert_images_to_png(...)`
- `convert_images_to_grayscale(...)`
- `resize_images(...)`
- `normalize_images(...)`
- `images_to_tensors(...)`

### `visualization.py`

- `draw_keypoints(...)`
- `draw_matches(...)`
- `draw_pck_thresholds(...)`

## Typical End-to-End Workflows

### Workflow A: full-image SuperPoint alignment

1. Annotate canonical keypoints on ideal template.
2. Prepare test images and optional `.npy` normalizations.
3. Run `run_pipeline.py`.
4. Annotate GT keypoints for test images.
5. Run `evaluator.py`.

### Workflow B: YOLO-cropped alignment + hex projection

1. Annotate cropped canonical keypoints on cropped ideal template.
2. Run `run_pipeline_cropped.py`.
3. (Optional) Evaluate with `cropped_eval.py`.
4. Annotate canonical hex centers (`hex_marker.py`).
5. Project hex centers (`projection/project_hex.py`).

## Troubleshooting

- `Model weights not found`
	- verify SuperPoint weight path in config constants.
- `No board detected` in cropped pipeline
	- lower detection threshold or verify YOLO weights path.
- `Homography failed`
	- ensure enough distinct matches and valid keypoint ordering.
- `shape mismatch gt/pred`
	- ensure same number/order of manually clicked keypoints between template and test annotations.
- Empty evaluation set
	- confirm `*_gt.npy`, `*_pred.npy` (and `*_bbox.npy` for cropped eval) exist with matching stems.

## Notes

- Most scripts use hardcoded config constants near the top. Edit those paths first for your local dataset layout.
- Some utility scripts are interactive OpenCV windows; run in a desktop session (not headless).
- `wrapper.py` offers a class-based interface and richer debug visualizations if you want to experiment beyond the minimal pipeline scripts.
