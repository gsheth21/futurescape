# Board Segmentation Module

This module contains code for detecting and segmenting board game boards using YOLO instance segmentation and Detectron2.

## 📋 Overview

The board segmentation pipeline consists of:
1. **Data Augmentation** (`polyseg_datasug.py`) - Generates augmented training data with polygon transformations
2. **YOLO Segmentation Training** (`yolo_seg.ipynb`) - Trains YOLOv8 segmentation model
3. **Detectron2 Segmentation** (`detectron2_segmentation.ipynb`) - Alternative using Detectron2 (advanced)
4. **Perspective Correction** - Extracts board corners and rectifies to 2D view

## 🎯 What is Board Segmentation?

Board segmentation identifies the **precise pixel-level outline** of the board in an image. This is more accurate than bounding boxes because it:
- Handles irregular board shapes
- Works with rotated/tilted boards
- Enables precise corner detection for perspective correction
- Provides exact board masks for background removal

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
cd board_segmentation
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
- `ultralytics==8.3.209` - YOLOv8 segmentation implementation
- `opencv-python==4.11.0.86` - Image processing
- `albumentations==2.0.8` - Advanced data augmentation
- `matplotlib==3.8.2` - Visualization
- `scikit-learn==1.6.1` - Data splitting
- `psutil` - Memory monitoring

## 📁 Directory Structure

```
board_segmentation/
├── yolo_seg.ipynb                 # Main YOLO segmentation notebook
├── detectron2_segmentation.ipynb  # Alternative Detectron2 approach
├── polyseg_datasug.py            # Polygon data augmentation
├── README.md                     # This file
│
├── board_segmentation_dataset/    # Dataset directory
│   ├── classes.txt               # Class names
│   ├── images/                   # Original images
│   ├── images_aug/               # Augmented images (generated)
│   ├── labels/                   # Polygon labels (YOLO format)
│   └── labels_aug/               # Augmented labels (generated)
│
├── yolo_segmentation_dataset/     # Train/val/test split (generated)
│   ├── dataset.yaml              # YOLO configuration
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── val/
│   └── test/
│
├── runs/                          # Training outputs (gitignored)
│   └── segment/
│       └── board_segmentation/
│
└── saved_models/                  # Trained models (gitignored)
    └── board_segmentation_v1.pt

Pre-trained models (download separately):
├── yolov8n-seg.pt                # YOLOv8 nano segmentation
└── yolov8m-seg.pt                # YOLOv8 medium segmentation
```

## 🔧 Usage

### Step 1: Prepare Annotations

Your dataset should be in YOLO segmentation format:
- **Images**: JPG or PNG format
- **Labels**: Text files with format `class x1 y1 x2 y2 ... xn yn` (normalized 0-1)

Place your data in:
- `board_segmentation_dataset/images/`
- `board_segmentation_dataset/labels/`

**YOLO Segmentation Label Format:**
```
class_id x1 y1 x2 y2 x3 y3 ... xn yn
```

Where:
- `class_id`: 0 (for board)
- `x1, y1, x2, y2, ...`: Polygon vertices in normalized coordinates (0-1)
- Minimum 3 points required for a valid polygon

### Step 2: Data Augmentation

```bash
python polyseg_datasug.py
```

This will:
- Apply perspective distortion
- Apply random rotations (±15°)
- Adjust brightness/contrast
- Apply Gaussian blur
- Generate 10 augmented versions per image
- Preserve polygon annotations through transformations

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

### Step 3: Train Model

Open and run `yolo_seg.ipynb` in Jupyter/VS Code.

**The notebook is organized into sections:**

1. **Diagnostic Cells** - Run first if kernel crashes (memory checks)
2. **Data Preparation** - Create train/val/test splits (70/20/10)
3. **Dataset Creation** - Generate YOLO dataset structure
4. **Visualization** - View sample annotations
5. **Model Training** - Train YOLOv8 segmentation model
6. **Evaluation** - Test model performance
7. **Inference** - Run predictions on test images
8. **Perspective Correction** - Extract corners and rectify boards

**Training Parameters:**
```python
EPOCHS = 40              # Training epochs (start with 10 for testing)
IMAGE_SIZE = 416         # Input image size
BATCH_SIZE = 8           # Batch size (reduce to 2-4 for 4GB VRAM)
LEARNING_RATE = 0.01     # Initial learning rate
```

**For RTX 3050 Ti (4GB VRAM):**
- Recommended `batch=4-8`
- Recommended `imgsz=320-416`
- Use `yolov8n-seg.pt` for faster training
- Use `yolov8m-seg.pt` for better accuracy
- Set `workers=0-2` to prevent crashes

### Step 4: Use Trained Model

```python
from ultralytics import YOLO
import cv2
import numpy as np

# Load trained model
model = YOLO("saved_models/board_segmentation_v1.pt")

# Run inference
results = model("path/to/test/image.jpg")

# Extract segmentation mask
result = results[0]
if result.masks is not None:
    mask = result.masks.data[0].cpu().numpy()
    
    # Resize to original image size
    img = cv2.imread("path/to/test/image.jpg")
    h, w = img.shape[:2]
    mask_resized = cv2.resize(mask, (w, h))
    
    # Apply mask
    masked_img = img.copy()
    masked_img[mask_resized < 0.5] = 0  # Black background
```

### Step 5: Perspective Correction

The notebook includes functions to:
1. **Extract corners from mask** - `extract_board_corners(mask)`
2. **Order corners properly** - `order_corners(pts)`
3. **Apply perspective transform** - `perform_perspective_correction(image, corners, output_size)`

```python
# Complete pipeline
original, rectified, corners, confidence = process_board_image_fixed(
    image_path, 
    model, 
    output_size=(800, 800),
    confidence_threshold=0.3,
    debug=True  # Show corner detection process
)
```

## 📊 Model Performance

### Expected Results

Good model performance:
- **mAP50 (Mask) > 0.85**: High segmentation accuracy
- **mAP50-95 (Mask) > 0.70**: Robust across IoU thresholds
- **Precision > 0.90**: Few false positives
- **Recall > 0.85**: Detects most boards

### Improving Performance

1. **More training data**: Add more labeled images
2. **More epochs**: Increase to 80-150 epochs
3. **Larger model**: Use yolov8m-seg or yolov8l-seg
4. **Better annotations**: Ensure polygon accuracy
5. **Tune augmentation**: Adjust transformation parameters

## ⚙️ Configuration

### GPU Setup

Verify GPU is available:
```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

### Memory Management

The notebook includes **ultra-safe mode** for low-memory systems:
- Reduced batch size (2-4)
- Smaller image size (320px)
- CPU-only training option
- Disabled features that consume RAM
- Frequent checkpointing

**If kernel crashes:**
1. Run diagnostic cells first
2. Close other applications
3. Use ultra-conservative settings
4. Consider cloud GPU (Google Colab)

### Dataset Split Ratios

Modify in the notebook:
```python
# Default: 70% train, 20% val, 10% test
train_names, temp_names = train_test_split(valid_pairs, test_size=0.3, random_state=42)
test_names, val_names = train_test_split(temp_names, test_size=0.333, random_state=42)
```

## 🛠️ Utility Functions

### Corner Detection from Mask

```python
def extract_board_corners(mask):
    """
    Extract 4 corner points from segmentation mask.
    
    Methods tried in order:
    1. Polygon approximation to quadrilateral
    2. Convex hull analysis
    3. Minimum rotated rectangle (fallback)
    """
    # Returns ordered corners: [top-left, top-right, bottom-right, bottom-left]
```

### Perspective Correction

```python
def perform_perspective_correction(image, corners, output_size=(800, 800)):
    """
    Apply perspective transformation to rectify board to 2D view.
    
    Args:
        image: Input image
        corners: 4 corner points
        output_size: Desired output dimensions
        
    Returns:
        rectified_image: Perspective-corrected board image
        transformation_matrix: Homography matrix
    """
```

## 🐛 Troubleshooting

### Common Issues

1. **Kernel crashes during training**
   - Run diagnostic cells first
   - Check available RAM (need 4GB+ free)
   - Use ultra-safe training mode
   - Reduce batch size to 2
   - Use CPU instead of GPU
   - Close other applications

2. **CUDA out of memory**
   - Reduce `batch` to 2-4
   - Reduce `imgsz` to 320
   - Use `yolov8n-seg` instead of `yolov8m-seg`
   - Set `cache=False`

3. **Poor segmentation quality**
   - Check annotation quality (polygon accuracy)
   - Increase training epochs (80-150)
   - Use larger model (yolov8m-seg)
   - Add more training data
   - Verify augmentation parameters

4. **Corner detection fails**
   - Mask quality may be poor (retrain with more data)
   - Try adjusting epsilon in polygon approximation
   - Use debug mode to visualize process
   - Check if mask has holes or disconnected regions

5. **Slow training on CPU**
   - Normal - CPU is 10-20x slower than GPU
   - Consider Google Colab with free GPU
   - Use smaller model (yolov8n-seg)
   - Reduce dataset size for testing

## 📝 Output Files

- `yolo_segmentation_dataset/dataset.yaml` - Dataset configuration
- `runs/segment/board_segmentation/weights/best.pt` - Best model
- `runs/segment/board_segmentation/weights/last.pt` - Last checkpoint
- `runs/segment/board_segmentation/results.png` - Training curves
- `saved_models/board_segmentation_v1.pt` - Exported model
- `saved_models/board_segmentation_v1_summary.json` - Training summary

## 🔄 Workflow Summary

```
Polygon Annotations → Data Augmentation → Train/Val/Test Split → 
YOLO Segmentation Training → Model Evaluation → Corner Extraction → 
Perspective Correction → Rectified Board
```

## 📚 Additional Resources

- [Ultralytics YOLOv8 Segmentation](https://docs.ultralytics.com/tasks/segment/)
- [Albumentations for Polygons](https://albumentations.ai/docs/getting_started/mask_augmentation/)
- [YOLO Segmentation Format](https://docs.ultralytics.com/datasets/segment/)
- [Perspective Transformation](https://docs.opencv.org/4.x/da/d6e/tutorial_py_geometric_transformations.html)

## 🤝 Contributing

When working on this module:
1. Maintain polygon annotation integrity during augmentation
2. Test on validation set before deploying
3. Document any changes to hyperparameters
4. Keep augmented data separate from originals
5. Version trained models with descriptive names

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
- RAM: 8GB+ (16GB+ recommended)
- Storage: 10GB+ for datasets and models

**Recommended for Stable Training:**
- RAM: 16GB+
- GPU: 6GB+ VRAM
- SSD for faster data loading

## 🔍 Detectron2 Alternative

The `detectron2_segmentation.ipynb` notebook provides an alternative approach using Facebook's Detectron2 framework. This is more advanced and requires additional setup but can provide better accuracy for complex scenarios.

## 📄 License

This project is part of the Board Game Detection system at NCSU Fall 2025.

## 👥 Authors

NCSU - Fall 2025 Board Game Project Team
