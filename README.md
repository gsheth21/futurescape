# FutureScape - Board Game Detection System

A comprehensive computer vision pipeline for detecting, segmenting, and analyzing board game boards using deep learning and geometric transformations.

## 📋 Project Overview

This project implements an end-to-end system for board game image analysis, from detecting the board in an image to mapping hexagonal grid coordinates. The pipeline uses YOLOv8 models for object detection, segmentation, and keypoint prediction, combined with classical computer vision techniques for perspective correction and homography transformation.

**Applications**:
- Automated game state recognition
- Digital board game analysis
- Augmented reality board game overlays
- Player move validation
- Game state recording and replay

## 🎯 System Architecture

The system consists of five integrated modules that work together to process board game images:

```
Input Image
    ↓
┌─────────────────────────────────────────────────────────────┐
│ 1. Board Detection (YOLOv8 Object Detection)                │
│    → Detects board bounding box                             │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. Board Segmentation (YOLOv8 Instance Segmentation)        │
│    → Extracts precise board mask and corners                │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. Board Keypoint Prediction (YOLOv8 Pose Estimation)       │
│    → Detects 9 hexagon center keypoints                     │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. Homography Transformation                                 │
│    → Warps ideal template to match board perspective        │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. Canonical 2D Template                                     │
│    → Overlays hexagonal grid on rectified board             │
└─────────────────────────────────────────────────────────────┘
    ↓
Output: Board with Mapped Hexagonal Grid
```

## 📁 Repository Structure

```
FutureScape/
├── README.md                          # This file
├── requirements.txt                   # Conda environment dependencies
│
├── board_detection/                   # Module 1: Board Detection
│   ├── README.md                     # Detection module documentation
│   ├── board_detection.ipynb         # Training notebook
│   ├── data_augmentation.py          # Bbox augmentation script
│   └── requirements.txt              # Module dependencies
│
├── board_keypoint_prediction/         # Module 2: Keypoint Prediction
│   ├── README.md                     # Keypoint module documentation
│   ├── board_keypoint.ipynb          # Training notebook
│   ├── detection_to_keypoint.ipynb   # Advanced pipeline
│   ├── convert_to_keypoint.py        # Label Studio converter
│   ├── keypoint_dataaug.py           # Keypoint augmentation
│   ├── find_coordinates.py           # Interactive coordinate tool
│   └── verify_augmentations.py       # Augmentation verification
│
├── board_segmentation/                # Module 3: Board Segmentation
│   ├── README.md                     # Segmentation module documentation
│   ├── yolo_seg.ipynb                # Main training notebook
│   ├── detectron2_segmentation.ipynb # Alternative approach
│   └── polyseg_datasug.py            # Polygon augmentation
│
├── canonical 2d template/             # Module 4: Reference Grid
│   ├── README.md                     # Template documentation
│   └── generate_hexagons.py          # Hexagonal grid generator
│
├── homography/                        # Module 5: Perspective Transform
│   ├── README.md                     # Homography documentation
│   ├── homography.ipynb              # Automated YOLO-based approach
│   └── homography_1.py               # Manual point selection
│
└── datasets/                          # Dataset directories (gitignored)
    ├── yolo_keypoint/
    ├── yolo_seg_dataset/
    └── ...
```

## 🚀 Getting Started

### Prerequisites

- **Python**: 3.9+
- **Conda**: For environment management
- **GPU**: NVIDIA GPU with 4GB+ VRAM (tested on RTX 3050 Ti)
  - Driver Version: 546.30
  - CUDA Version: 12.3
- **RAM**: 8GB+ (16GB recommended)
- **Storage**: 10GB+ for datasets and models

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/gpsheth_ncstate/FutureScape.git
cd FutureScape
```

2. **Create conda environment:**
```bash
conda create -n futurescape python=3.9
conda activate futurescape
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

### Key Dependencies

```
torch==2.8.0              # PyTorch with CUDA 12 support
torchvision==0.23.0       # Computer vision models
ultralytics==8.3.209      # YOLOv8 implementation
opencv-python==4.11.0.86  # Image processing
albumentations==2.0.8     # Data augmentation
numpy==1.24.3             # Numerical computing
matplotlib==3.8.2         # Visualization
scikit-learn==1.6.1       # ML utilities
```

## 📖 Module Documentation

Each module has its own comprehensive README with detailed usage instructions:

### 1. Board Detection
**Purpose**: Detect board bounding boxes in images

**Method**: YOLOv8 object detection

**Key Features**:
- Bounding box augmentation
- 80/20 train/val split
- GPU-optimized training

📚 [Read More →](board_detection/README.md)

### 2. Board Keypoint Prediction
**Purpose**: Detect 9 hexagon center keypoints for perspective mapping

**Method**: YOLOv8 pose estimation

**Key Features**:
- Keypoint-preserving augmentation
- Label Studio integration
- Interactive coordinate finder
- Advanced detection-to-keypoint pipeline

📚 [Read More →](board_keypoint_prediction/README.md)

### 3. Board Segmentation
**Purpose**: Segment board with pixel-level precision

**Method**: YOLOv8 instance segmentation + Detectron2

**Key Features**:
- Polygon segmentation
- Corner extraction
- Perspective correction
- Memory-safe training for 4GB GPUs

📚 [Read More →](board_segmentation/README.md)

### 4. Canonical 2D Template
**Purpose**: Generate reference hexagonal grid template

**Method**: Geometric computation from reference points

**Key Features**:
- Regular hexagon grid generation
- "odd-q" vertical layout
- Multiple export formats (NumPy, CSV, JSON)
- Visualization tools

📚 [Read More →](canonical 2d template/README.md)

### 5. Homography
**Purpose**: Warp ideal template to match real board perspective

**Method**: Homography transformation (manual + automated)

**Key Features**:
- Manual point selection approach
- Automated YOLO-based corner detection
- Overlay visualization
- Quality validation

📚 [Read More →](homography/README.md)

## 🔧 Complete Pipeline Usage

### Quick Start: Process a Single Image

```python
from ultralytics import YOLO
import cv2
import numpy as np

# 1. Load models
detector = YOLO("board_detection/saved_models/board_detector_v1.pt")
segmenter = YOLO("board_segmentation/saved_models/board_segmentation_v1.pt")
keypointer = YOLO("board_keypoint_prediction/saved_models/board_keypoint_v1.pt")

# 2. Load image
img = cv2.imread("test_image.jpg")

# 3. Detect board
det_results = detector(img)
bbox = det_results[0].boxes.xyxy[0].cpu().numpy()
x1, y1, x2, y2 = bbox.astype(int)
board_crop = img[y1:y2, x1:x2]

# 4. Segment board
seg_results = segmenter(board_crop)
mask = seg_results[0].masks.data[0].cpu().numpy()

# 5. Extract corners
corners = extract_corners_from_mask(mask)

# 6. Detect keypoints
kp_results = keypointer(board_crop)
keypoints = kp_results[0].keypoints.xy[0].cpu().numpy()

# 7. Apply homography
ideal_template = cv2.imread("ideal_image.png")
H = cv2.findHomography(ideal_corners, corners)
warped = cv2.warpPerspective(ideal_template, H, (board_crop.shape[1], board_crop.shape[0]))

# 8. Overlay result
overlay = cv2.addWeighted(board_crop, 0.5, warped, 0.5, 0)
cv2.imshow("Result", overlay)
cv2.waitKey(0)
```

### Training Pipeline

Follow these steps to train all models from scratch:

1. **Prepare datasets** using Label Studio or similar annotation tool
2. **Train detection model** (1-2 hours on RTX 3050 Ti):
   ```bash
   cd board_detection
   jupyter notebook board_detection.ipynb
   ```

3. **Train segmentation model** (2-3 hours):
   ```bash
   cd board_segmentation
   jupyter notebook yolo_seg.ipynb
   ```

4. **Train keypoint model** (1-2 hours):
   ```bash
   cd board_keypoint_prediction
   jupyter notebook board_keypoint.ipynb
   ```

5. **Generate canonical template**:
   ```bash
   cd "canonical 2d template"
   python generate_hexagons.py
   ```

6. **Test homography transformation**:
   ```bash
   cd homography
   python homography_1.py  # Manual method
   # OR
   jupyter notebook homography.ipynb  # Automated method
   ```

## 📊 Performance Benchmarks

### Hardware: NVIDIA RTX 3050 Ti (4GB VRAM)

| Module | Training Time | Inference Speed | mAP50 | Notes |
|--------|---------------|-----------------|-------|-------|
| Board Detection | 1-2 hours (40 epochs) | ~30 FPS | 0.95+ | YOLOv8n, batch=8 |
| Board Segmentation | 2-3 hours (40 epochs) | ~20 FPS | 0.85+ | YOLOv8n-seg, batch=4 |
| Keypoint Prediction | 1-2 hours (40 epochs) | ~25 FPS | 0.90+ | YOLOv8n-pose, batch=8 |

### Model Sizes

- Detection model: ~6 MB
- Segmentation model: ~7 MB  
- Keypoint model: ~6 MB
- **Total**: ~19 MB (very deployment-friendly)

## 🎨 Data Augmentation

All modules use sophisticated data augmentation:

- **Geometric**: Rotation (±15°), perspective distortion, scaling
- **Photometric**: Brightness/contrast, blur, noise
- **Spatial**: Crop, flip (disabled for keypoints)
- **Annotation-preserving**: Transforms applied to both images and labels

**Augmentation Ratios**:
- Detection: 10× augmentation per image
- Segmentation: 10× with polygon preservation
- Keypoints: 10× with coordinate transformation

## 🔍 Quality Validation

### Visual Inspection Tools

Each module includes visualization utilities:

```python
# Detection
visualize_predictions(img, boxes, labels)

# Segmentation  
visualize_mask_and_polygon(img, mask, corners)

# Keypoints
draw_keypoints(img, keypoints, skeleton)

# Homography
create_overlay(original, warped, alpha=0.5)
```

### Automated Metrics

- **Detection**: mAP, precision, recall
- **Segmentation**: mAP (mask), IoU
- **Keypoints**: OKS (Object Keypoint Similarity), PCK
- **Homography**: Reprojection error

## 🐛 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```python
# Reduce batch size
BATCH_SIZE = 2  # Instead of 8

# Reduce image size
IMAGE_SIZE = 320  # Instead of 640

# Clear cache
import torch
torch.cuda.empty_cache()
```

**2. Kernel Crashes**
- Close other applications
- Run diagnostic cells in notebooks
- Use CPU instead of GPU (slower but stable)
- Enable ultra-safe mode in training notebooks

**3. Poor Model Performance**
- Increase training epochs (40 → 80)
- Use larger models (yolov8n → yolov8m)
- Add more training data
- Check annotation quality
- Adjust augmentation parameters

**4. Installation Issues**
```bash
# PyTorch with CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# If opencv issues
pip uninstall opencv-python opencv-python-headless
pip install opencv-python==4.11.0.86
```

## 🔄 Integration Examples

### With Web API (FastAPI)

```python
from fastapi import FastAPI, UploadFile
from ultralytics import YOLO
import cv2
import numpy as np

app = FastAPI()
model = YOLO("board_segmentation/saved_models/board_segmentation_v1.pt")

@app.post("/detect-board")
async def detect_board(file: UploadFile):
    # Read uploaded image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Run segmentation
    results = model(img)
    mask = results[0].masks.data[0].cpu().numpy()
    
    # Extract corners
    corners = extract_corners(mask)
    
    return {"corners": corners.tolist()}
```

### With Real-time Video

```python
import cv2
from ultralytics import YOLO

model = YOLO("board_detection/saved_models/board_detector_v1.pt")
cap = cv2.VideoCapture(0)  # Webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run detection
    results = model(frame, stream=True)
    
    # Draw results
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    cv2.imshow("Board Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## 📚 Additional Resources

### Documentation
- [YOLOv8 Official Docs](https://docs.ultralytics.com/)
- [OpenCV Documentation](https://docs.opencv.org/)
- [Albumentations Docs](https://albumentations.ai/docs/)

### Research Papers
- **YOLOv8**: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- **Homography**: Hartley & Zisserman, "Multiple View Geometry"
- **Instance Segmentation**: He et al., "Mask R-CNN"

### Tutorials
- [Hexagonal Grid Math](https://www.redblobgames.com/grids/hexagons/)
- [Perspective Transformation](https://docs.opencv.org/4.x/da/d6e/tutorial_py_geometric_transformations.html)
- [YOLO Custom Training](https://docs.ultralytics.com/modes/train/)

## 🤝 Contributing

### Code Standards
- Follow PEP 8 for Python code
- Add docstrings to all functions
- Include type hints where appropriate
- Write unit tests for new features
- Update README when adding features

### Pull Request Process
1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

### Development Workflow
```bash
# Setup development environment
conda create -n futurescape-dev python=3.9
conda activate futurescape-dev
pip install -r requirements.txt

# Run tests
pytest tests/

# Check code style
flake8 .
```

## 📄 License

This project is part of the NCSU Fall 2025 Board Game Detection project.

## 👥 Authors

**NCSU - Fall 2025 Board Game Project Team**

## 🙏 Acknowledgments

- **Ultralytics** for YOLOv8 framework
- **OpenCV** community for computer vision tools
- **Albumentations** team for augmentation library
- NCSU faculty and advisors

## 📞 Contact

For questions, issues, or collaboration:
- **GitHub Issues**: [Create an issue](https://github.com/gpsheth_ncstate/FutureScape/issues)
- **Email**: Contact through GitHub profile

## 🗺️ Roadmap

### Current Features (v1.0)
- ✅ Board detection with YOLOv8
- ✅ Instance segmentation with corner extraction
- ✅ Keypoint prediction for 9 hex centers
- ✅ Homography transformation
- ✅ Canonical template generation

### Planned Features (v2.0)
- 🔲 Game piece detection and classification
- 🔲 Automated game state recognition
- 🔲 Multi-board support (different board types)
- 🔲 Real-time video processing optimization
- 🔲 Mobile deployment (ONNX/TensorRT)
- 🔲 Web interface for easy testing
- 🔲 Dataset expansion and public release

### Future Research
- 🔲 3D board reconstruction
- 🔲 Player action tracking
- 🔲 Game rule validation
- 🔲 AR overlay integration

---

**⭐ Star this repository if you find it useful!**

**🔗 Share with others working on board game digitization or computer vision projects.**
