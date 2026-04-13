# Board Keypoint Prediction Module

This module contains code for detecting and predicting keypoint locations (hex centers) on board game boards using YOLO Pose estimation.

## 📋 Overview

The board keypoint prediction pipeline consists of several components:
1. **Label Conversion** (`convert_to_keypoint.py`) - Converts Label Studio annotations to YOLO pose format
2. **Data Augmentation** (`keypoint_dataaug.py`) - Generates augmented training data with keypoint transformations
3. **Model Training** (`board_keypoint.ipynb` / `detection_to_keypoint.ipynb`) - Trains YOLOv8-pose model
4. **Utilities** - Helper scripts for coordinate finding and verification

## 🎯 What are Keypoints?

In this project, keypoints represent the **9 hex centers** on the board game board. Each keypoint has:
- **X, Y coordinates**: Location in the image (normalized 0-1)
- **Visibility flag**: 0 = not labeled, 1 = occluded, 2 = visible

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- Conda environment manager
- NVIDIA GPU with CUDA support (tested on RTX 3050 Ti with 4GB VRAM)
  - Driver Version: 546.30
  - CUDA Version: 12.3

### Installation

1. **Clone the repository:**
```bash
git clone <your-repo-url>
cd board_keypoint_prediction
```

2. **Create and activate conda environment:**
```bash
conda create -n futurescape python=3.9
conda activate futurescape
```

3. **Install required packages:**
```bash
pip install -r ../requirements.txt
```

### Key Dependencies

- `torch==2.8.0` - PyTorch with CUDA 12 support
- `torchvision==0.23.0` - Computer vision utilities
- `ultralytics==8.3.209` - YOLOv8 pose implementation
- `opencv-python==4.11.0.86` - Image processing
- `albumentations==2.0.8` - Advanced data augmentation
- `matplotlib==3.8.2` - Visualization
- `scikit-learn==1.6.1` - Metrics evaluation

## 📁 Directory Structure

```
board_keypoint_prediction/
├── board_keypoint.ipynb           # Simple training notebook
├── detection_to_keypoint.ipynb    # Full pipeline with evaluation
├── convert_to_keypoint.py         # Label Studio → YOLO converter
├── keypoint_dataaug.py           # Data augmentation script
├── find_coordinates.py            # Interactive coordinate finder
├── verify_augmentations.py        # Augmentation verification
├── README.md                      # This file
│
├── yolo_keypoint_dataset/         # Dataset directory
│   ├── classes.txt               # Class names
│   ├── notes.json                # Dataset metadata
│   ├── yolo_keypoint_dataset.json # Label Studio export
│   ├── images/                   # Original images
│   ├── images_aug/               # Augmented images (generated)
│   ├── labels/                   # YOLO pose labels (generated)
│   └── labels_aug/               # Augmented labels (generated)
│
├── yolo_pose_dataset/             # Train/val/test split (generated)
│   ├── dataset.yaml              # YOLO configuration
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── val/
│   └── test/
│
├── runs/                          # Training outputs (gitignored)
│   └── pose/
│       └── board_pose/
│
└── saved_models/                  # Trained models (gitignored)
    └── board_pose_v1.pt

Pre-trained models (download separately):
├── yolo11n.pt                     # YOLO11 nano
├── yolov8n-pose.pt               # YOLOv8 nano pose
└── yolov8m-pose.pt               # YOLOv8 medium pose
```

## 🔧 Usage

### Step 1: Prepare Annotations (Label Studio)

1. **Export annotations from Label Studio** in JSON format
2. Save as `yolo_keypoint_dataset/yolo_keypoint_dataset.json`
3. Ensure keypoints are labeled as: `Hex1`, `Hex2`, ..., `Hex9`

**Label Studio Configuration:**
- Use keypoint annotation tool
- Label 9 hex centers on each board
- Mark occluded points with "occluded" flag

### Step 2: Convert Labels to YOLO Format

```bash
python convert_to_keypoint.py
```

This will:
- Read Label Studio JSON annotations
- Convert to YOLO pose format
- Create label files in `yolo_keypoint_dataset/labels/`

**YOLO Pose Label Format:**
```
class x_center y_center width height kp1_x kp1_y kp1_v kp2_x kp2_y kp2_v ... kp9_x kp9_y kp9_v
```

Where:
- `class`: 0 (board)
- `x_center, y_center, width, height`: Bounding box (normalized 0-1)
- `kp_x, kp_y`: Keypoint coordinates (normalized 0-1)
- `kp_v`: Visibility (0=not labeled, 1=occluded, 2=visible)

### Step 3: Data Augmentation

```bash
python keypoint_dataaug.py
```

This will:
- Apply perspective distortion
- Apply random rotations (±15°)
- Adjust brightness/contrast
- Apply Gaussian blur
- Generate 10 augmented versions per image
- Preserve keypoint annotations through transformations

**Augmentation Configuration:**
```python
N_AUGS = 10  # Number of augmentations per image

transform = A.Compose([
    A.Perspective(scale=(0.05, 0.1), p=0.3),
    A.Rotate(limit=15, p=0.7),
    A.RandomBrightnessContrast(p=0.3),
    A.GaussianBlur(blur_limit=3, p=0.2)
])
```

### Step 4: Train Model

**Option A: Simple Training** (`board_keypoint.ipynb`)

1. Open `board_keypoint.ipynb` in Jupyter/VS Code
2. Run cells sequentially:
   - Setup paths and configuration
   - Create train/val/test splits (70/20/10)
   - Generate dataset.yaml
   - Train YOLOv8-pose model
   - Visualize results

**Option B: Full Pipeline** (`detection_to_keypoint.ipynb`)

Includes additional features:
- Bbox calculation from keypoints
- Advanced augmentation with bbox updates
- Comprehensive evaluation metrics
- Side-by-side GT vs prediction visualization

**Training Parameters:**
```python
model = YOLO('yolov8m-pose.pt')  # or yolov8n-pose.pt for faster training

results = model.train(
    data='yolo_pose_dataset/dataset.yaml',
    epochs=80,              # Number of training epochs
    imgsz=416,              # Input image size
    batch=8,                # Batch size (adjust for GPU memory)
    device='cuda',          # Use GPU (or 'cpu')
    workers=2,              # Data loading workers
    single_cls=True,        # Single class detection
    name="board_pose"
)
```

### Step 5: Evaluate Model

The `detection_to_keypoint.ipynb` notebook includes comprehensive evaluation:

**Metrics Calculated:**
- **MAE** (Mean Absolute Error): Average distance error
- **RMSE** (Root Mean Square Error): Overall prediction accuracy
- **Per-keypoint metrics**: Individual hex center accuracy
- **Confidence scores**: Model certainty for each prediction
- **Success rate**: Percentage within acceptable threshold

**Example Output:**
```
📊 OVERALL METRICS:
  mae................................. 0.012345
  rmse................................ 0.015678
  success_rate........................ 0.945000

📍 PER-KEYPOINT METRICS:
  Hex1:
    MAE (pixels):        0.011234
    RMSE (pixels):       0.014567
    Avg Confidence:      0.9567
```

### Step 6: Use Trained Model

```python
from ultralytics import YOLO
import cv2

# Load trained model
model = YOLO("saved_models/board_pose_v1.pt")

# Run inference
results = model("path/to/test/image.jpg")

# Extract keypoints
result = results[0]
if result.keypoints is not None:
    keypoints = result.keypoints.xy[0]  # (9, 2) array
    confidences = result.keypoints.conf[0]  # (9,) array
    
    for i, (kp, conf) in enumerate(zip(keypoints, confidences)):
        x, y = kp.cpu().numpy()
        print(f"Hex{i+1}: ({x:.1f}, {y:.1f}), confidence: {conf:.3f}")
```

## 🛠️ Utility Scripts

### find_coordinates.py

Interactive tool to manually find hex center coordinates:

```bash
python find_coordinates.py
```

1. Update image path in the script
2. Click on each hex center
3. Coordinates are printed and saved

### verify_augmentations.py

Verify augmentation quality by visualizing bboxes:

```bash
python verify_augmentations.py
```

Creates preview images in `Dataset/preview/` with bounding boxes drawn.

## ⚙️ Configuration

### GPU Setup

Verify GPU is available:
```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

### For RTX 3050 Ti (4GB VRAM):
- Recommended `batch=4-8`
- Recommended `imgsz=416`
- Use `yolov8n-pose.pt` for faster training
- Use `yolov8m-pose.pt` for better accuracy

### Dataset Split Ratios

Modify in training notebooks:
```python
n_train = int(0.7 * n_total)  # 70% training
n_val = int(0.2 * n_total)    # 20% validation
n_test = n_total - n_train - n_val  # 10% test
```

## 📊 Model Performance

### Expected Results

Good model performance:
- **MAE < 0.02**: Keypoints within 2% of image size
- **Success rate > 90%**: Most predictions accurate
- **Confidence > 0.8**: High prediction certainty

### Improving Performance

1. **More training data**: Add more labeled images
2. **More epochs**: Increase to 100-150 epochs
3. **Larger model**: Use yolov8m-pose or yolov8l-pose
4. **Better augmentation**: Tune augmentation parameters
5. **Clean labels**: Verify annotation quality

## 🐛 Troubleshooting

### Common Issues

1. **CUDA out of memory**
   - Reduce `batch` size to 2-4
   - Reduce `imgsz` to 320
   - Use smaller model (yolov8n-pose)

2. **Low keypoint accuracy**
   - Check label quality in Label Studio
   - Increase epochs
   - Add more training data
   - Verify keypoint order (Hex1-Hex9)

3. **Keypoints not detected**
   - Lower confidence threshold
   - Check if bbox covers keypoints
   - Verify visibility flags in labels

4. **Augmentation errors**
   - Check keypoint coordinates are normalized (0-1)
   - Verify all 9 keypoints are labeled
   - Check for NaN or inf values

5. **Training crashes**
   - Reduce workers to 0
   - Check dataset paths
   - Verify label file format

## 📝 Output Files

- `yolo_pose_dataset/dataset.yaml` - Dataset configuration
- `runs/pose/board_pose/weights/best.pt` - Best model checkpoint
- `runs/pose/board_pose/weights/last.pt` - Last checkpoint
- `runs/pose/board_pose/results.png` - Training curves
- `saved_models/board_pose_v1.pt` - Exported model

## 🔄 Workflow Summary

```
Label Studio Export → Convert to YOLO Format → Data Augmentation → 
Train/Val/Test Split → YOLO Pose Training → Model Evaluation → Trained Model
```

## 📚 Additional Resources

- [Ultralytics YOLOv8 Pose Documentation](https://docs.ultralytics.com/tasks/pose/)
- [Albumentations for Keypoints](https://albumentations.ai/docs/getting_started/keypoints_augmentation/)
- [Label Studio Keypoint Annotation](https://labelstud.io/guide/keypoint.html)
- [YOLO Pose Format Specification](https://docs.ultralytics.com/datasets/pose/)

## 🤝 Contributing

When working on this module:
1. Maintain keypoint ordering (Hex1-Hex9)
2. Preserve visibility flags during augmentation
3. Test on validation set before deploying
4. Document any changes to hyperparameters
5. Keep augmented data separate from originals

## 💻 System Requirements

**Tested Configuration:**
- OS: Linux (WSL2 on Windows)
- GPU: NVIDIA GeForce RTX 3050 Ti (4GB VRAM)
- CUDA: 12.3
- Driver: 546.30
- Python: 3.9
- Environment: Conda (futurescape)

**Minimum Requirements:**
- GPU: 4GB VRAM (or CPU with patience)
- RAM: 8GB+
- Storage: 5GB+ for datasets and models

## 📄 License

This project is part of the Board Game Detection system at NCSU Fall 2025.

## 👥 Authors

NCSU - Fall 2025 Board Game Project Team
