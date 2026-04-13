# Piece Detection

This folder contains tools for:

1. Visualizing YOLO labels on images.
2. Training and running a red piece detector using YOLOv8.

## Folder Structure

- [data_visualization](data_visualization)
	- [labels_viz.py](data_visualization/labels_viz.py): visualize labels for a single image or batch process a full dataset.
- [red_piece_detection](red_piece_detection)
	- [detection.ipynb](red_piece_detection/detection.ipynb): end-to-end notebook for dataset prep, board-first cropping, training, validation, and inference.
	- [saved_models](red_piece_detection/saved_models): exported trained model weights.
	- [runs](red_piece_detection/runs): YOLO training and validation outputs.

## 1) Label Visualization

Script: [data_visualization/labels_viz.py](data_visualization/labels_viz.py)

### Dataset Format

Expected YOLO layout:

```text
dataset_dir/
	classes.txt
	notes.json
	images/
		*.jpg | *.png
	labels/
		*.txt
```

### Single Image (Show Only)

Displays one image with its bounding boxes and class labels.

```bash
python data_visualization/labels_viz.py show <image_path> <label_path> <classes_txt_path>
```

Example:

```bash
python data_visualization/labels_viz.py show \
	"../dataset/Red-Less_Resilient_Labels/Transforming futures workshop 2024/images/1a440db2-4-2060.jpg" \
	"../dataset/Red-Less_Resilient_Labels/Transforming futures workshop 2024/labels/1a440db2-4-2060.txt" \
	"../dataset/Red-Less_Resilient_Labels/Transforming futures workshop 2024/classes.txt"
```

### Batch Mode (Save Annotated Images)

Runs through every image in a dataset and saves annotated outputs.

```bash
python data_visualization/labels_viz.py batch <dataset_dir> <output_dir>
```

## 2) Red Piece Detection Pipeline

Notebook: [red_piece_detection/detection.ipynb](red_piece_detection/detection.ipynb)

The notebook pipeline is:

1. Load source dataset.
2. Detect board region and crop images to the board.
3. Remap piece labels to crop coordinates.
4. Split into train/val/test.
5. Train YOLOv8 red piece detector.
6. Validate and inspect metrics.
7. Run two-stage inference:
	 - stage A: board detector,
	 - stage B: piece detector inside board crop,
	 - map detections back to original image pixel coordinates.
8. Save model to [red_piece_detection/saved_models/red_piece_detector.pt](red_piece_detection/saved_models/red_piece_detector.pt).

## Setup

From the project root:

```bash
pip install -r requirements.txt
```

If needed for visualization only:

```bash
pip install opencv-python
```

## Outputs You Should Expect

- Trained weights under [red_piece_detection/runs/detect](red_piece_detection/runs/detect).
- Final exported model under [red_piece_detection/saved_models](red_piece_detection/saved_models).
- Validation artifacts like PR/F1/confusion plots in YOLO run directories.

## Notes

- The two-stage approach (board first, then pieces) reduces false detections outside the board.
- Prediction outputs include bounding boxes for all detected pieces; in two-stage mode, convert crop coordinates back to original image coordinates using crop offsets.
