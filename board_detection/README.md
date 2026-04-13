# Board Detection Module

This module contains code for detecting board game boards in images using YOLO (You Only Look Once) object detection.

## 📋 Overview

The board detection pipeline consists of two main components:
1. **Data Augmentation** (`data_augmentation.py`) - Generates augmented training data from labeled images
2. **Model Training** (`board_detection.ipynb`) - Trains a YOLOv8 model to detect boards in images

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Conda environment manager
- NVIDIA GPU with CUDA support (tested on RTX 3050 Ti with 4GB VRAM)
  - Driver Version: 546.30
  - CUDA Version: 12.3

### Installation

1. **Clone the repository:**
```bash
git clone <your-repo-url>
cd board_detection
```

2. **Create and activate a conda environment:**
```bash
conda create -n futurescape python=3.9
conda activate futurescape
```

3. **Install required packages:**
```bash
pip install -r requirements.txt
```

Alternatively, install from the parent directory's requirements:
```bash
pip install -r ../requirements.txt
```

### Key Dependencies

- `torch==2.8.0` - PyTorch deep learning framework with CUDA 12 support
- `torchvision==0.23.0` - Computer vision utilities
- `ultralytics==8.3.209` - YOLOv8 implementation
- `opencv-python==4.11.0.86` - Image processing
- `albumentations==2.0.8` - Data augmentation library
- `PyYAML==6.0.3` - YAML configuration handling
- `numpy==1.24.3` - Numerical computing
- `matplotlib==3.8.2` - Visualization

For a complete list of dependencies, see `requirements.txt` in the parent directory.

## 📁 Directory Structure

```
board_detection/
├── board_detection.ipynb    # Main training notebook
├── data_augmentation.py     # Data augmentation script
└── README.md               # This file

datasets/
└── bounding_box_dataset/   # Your dataset directory
    ├── images/             # Original images
    ├── labels/             # YOLO format labels
    ├── images_aug/         # Augmented images (generated)
    ├── labels_aug/         # Augmented labels (generated)
    └── yolo_dataset/       # Train/val split (generated)
        ├── data.yaml
        ├── train/
        │   ├── images/
        │   └── labels/
        └── val/
            ├── images/
            └── labels/

saved_models/               # Trained model weights
└── board_detector_v1.pt
```

## 🔧 Usage

### Step 1: Prepare Your Dataset

Your dataset should be in YOLO format:
- **Images**: JPG or PNG format
- **Labels**: Text files with format `class x_center y_center width height` (normalized 0-1)

Place your data in:
- `datasets/bounding_box_dataset/images/`
- `datasets/bounding_box_dataset/labels/`

### Step 2: Data Augmentation

Run the augmentation script to generate additional training data:

```bash
python data_augmentation.py
```

This will:
- Apply random horizontal/vertical flips
- Apply small rotations (±15°)
- Generate 10 augmented versions per image
- Save augmented data to `images_aug/` and `labels_aug/`

**Configuration:**
- Modify `N_AUGS` variable to change number of augmentations per image
- Adjust augmentation parameters in the `transform` pipeline

### Step 3: Train the Model

Open and run `board_detection.ipynb` in Jupyter or VS Code:

1. **Setup paths** - Verifies dataset structure
2. **Train/Val split** - Splits data into 80% training, 20% validation
3. **Create data.yaml** - Generates YOLO configuration file
4. **Check GPU** - Detects if CUDA is available
5. **Train model** - Trains YOLOv8n on your dataset
6. **Validate** - Evaluates model performance
7. **Save model** - Exports trained weights to `saved_models/`

### Step 4: Use Trained Model

```python
from ultralytics import YOLO
from pathlib import Path

# Load trained model
model = YOLO("saved_models/board_detector_v1.pt")

# Run inference
results = model("path/to/test/image.jpg")

# Process results
for result in results:
    boxes = result.boxes  # Bounding boxes
    print(f"Detected {len(boxes)} boards")
```

## ⚙️ Configuration

### GPU Setup

The project is configured to use NVIDIA GPU for training. Verify your GPU is detected:

```bash
nvidia-smi
```

Expected output should show your GPU (e.g., RTX 3050 Ti with 4GB VRAM).

### Training Parameters

Edit these in the notebook to customize training:

```python
model.train(
    data="path/to/data.yaml",
    epochs=20,              # Number of training epochs
    imgsz=416,              # Input image size (adjust based on GPU memory)
    batch=4,                # Batch size (for 4GB VRAM, use 4-8)
    device=0,               # Use GPU 0 (set to "cpu" for CPU training)
    workers=0,              # Number of data loading workers
    cache=True,             # Cache images for faster training
    patience=5,             # Early stopping patience
)
```

**For RTX 3050 Ti (4GB VRAM):**
- Recommended `batch=4-8`
- Recommended `imgsz=416` or `640`
- Use `device=0` to leverage GPU acceleration

### Augmentation Parameters

Edit these in `data_augmentation.py`:

```python
N_AUGS = 10  # Augmentations per image

transform = A.Compose([
    A.HorizontalFlip(p=0.5),     # 50% chance horizontal flip
    A.VerticalFlip(p=0.5),       # 50% chance vertical flip
    A.Rotate(limit=15, p=0.7)    # ±15° rotation, 70% chance
])
```

## 📊 Model Performance

After training, check:
- **Precision/Recall**: In validation metrics
- **mAP (mean Average Precision)**: Overall detection accuracy
- **Visual results**: In `runs/detect/board-detector/`

## 🐛 Troubleshooting

### Common Issues

1. **CUDA out of memory (4GB VRAM)**
   - Reduce `batch` size to 2-4
   - Reduce `imgsz` (e.g., 320 instead of 416)
   - Use `yolov8n.pt` (nano) instead of larger models
   - Close other GPU-intensive applications

2. **GPU not detected**
   - Verify CUDA installation: `nvidia-smi`
   - Check PyTorch CUDA availability: `python -c "import torch; print(torch.cuda.is_available())"`
   - Ensure CUDA version matches PyTorch installation (CUDA 12.3)

3. **Dataset not found**
   - Verify paths in notebook match your directory structure
   - Check that images and labels have matching filenames
   - Ensure you're in the correct working directory

4. **Poor detection results**
   - Increase `epochs` (e.g., 50-100)
   - Add more training data and augmentations
   - Try a larger model if GPU memory allows (yolov8s.pt)
   - Check label quality and format

5. **Slow training on CPU**
   - Make sure GPU is being used: check `device=0` parameter
   - Verify CUDA is available in your environment
   - Training on CPU is ~10-20x slower than GPU

## 📝 Output Files

- `saved_models/board_detector_v1.pt` - Best model weights
- `runs/detect/board-detector/` - Training logs and visualizations
- `runs/detect/board-detector/weights/best.pt` - Best checkpoint
- `runs/detect/board-detector/weights/last.pt` - Last checkpoint

## 🔄 Workflow Summary

```
Original Images → Data Augmentation → Train/Val Split → YOLO Training (GPU) → Trained Model
```

## 💻 System Requirements

**Tested Configuration:**
- OS: Linux (WSL2 on Windows)
- GPU: NVIDIA GeForce RTX 3050 Ti (4GB VRAM)
- CUDA: 12.3
- Driver: 546.30
- Python: 3.9
- Environment: Conda (futurescape)

## 📚 Additional Resources

- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
- [Albumentations Documentation](https://albumentations.ai/)
- [YOLO Label Format](https://docs.ultralytics.com/datasets/detect/)

## 🤝 Contributing

When working on this module:
1. Keep augmented data separate from original data
2. Version your trained models with descriptive names
3. Document any changes to hyperparameters
4. Test on validation set before deploying

## 📄 License

This project is part of the Board Game Detection system at NCSU Fall 2025.

## 👥 Authors

NCSU - Fall 2025 Board Game Project Team
