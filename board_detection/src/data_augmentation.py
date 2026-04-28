"""
Data Augmentation Script for Board Detection

This script performs data augmentation on a labeled image dataset to increase
training data variety and improve model generalization. It uses the Albumentations
library to apply geometric transformations while preserving bounding box annotations.

Author: NCSU - Fall 2025 Board Game Project
"""

import os
import cv2
import albumentations as A
import random
from pathlib import Path
from dotenv import load_dotenv

# ============================================================================
# CONFIGURATION
# ============================================================================

# Load .env from the board_detection/ directory (one level up from src/)
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

# Dataset directory paths (set via .env)
DATASET_DIR = Path(os.environ["AUG_DATASET_DIR"])
IMG_DIR     = DATASET_DIR / "images"
LBL_DIR     = DATASET_DIR / "labels"

# Output directories for augmented data
OUT_IMG_DIR = DATASET_DIR / "images_aug"
OUT_LBL_DIR = DATASET_DIR / "labels_aug"

# Create output directories if they don't exist
os.makedirs(OUT_IMG_DIR, exist_ok=True)
os.makedirs(OUT_LBL_DIR, exist_ok=True)

# ============================================================================
# INITIALIZATION & VALIDATION
# ============================================================================

# Print diagnostic information
print(f"Looking for images in: {IMG_DIR}")
print(f"Directory exists: {os.path.exists(IMG_DIR)}")

# ============================================================================
# AUGMENTATION PIPELINE
# ============================================================================

# Define augmentation transformations using Albumentations
# These transformations are applied randomly to create diverse training samples
transform = A.Compose([
    A.HorizontalFlip(p=0.5),  # 50% chance to flip image horizontally
    A.VerticalFlip(p=0.5),    # 50% chance to flip image vertically
    A.Rotate(
        limit=15,              # Rotate by ±15 degrees
        border_mode=cv2.BORDER_CONSTANT,  # Fill borders with constant value
        p=0.7                  # 70% chance to apply rotation
    )
], bbox_params=A.BboxParams(
    format="yolo",             # YOLO format: [x_center, y_center, width, height]
    label_fields=["class_labels"],  # Field name for class labels
    min_visibility=0.5         # Keep boxes only if >50% visible after transform
))

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def read_yolo_labels(label_path):
    """
    Read YOLO format label file and extract bounding boxes and class labels.
    
    YOLO format: Each line contains [class_id x_center y_center width height]
    All coordinates are normalized (0-1 range)
    
    Args:
        label_path (str): Path to the YOLO .txt label file
        
    Returns:
        tuple: (boxes, labels) where:
            - boxes: List of [x_center, y_center, width, height] for each object
            - labels: List of class IDs (integers)
    """
    boxes, labels = [], []
    
    # Check if label file exists
    if not os.path.exists(label_path):
        print(f"Label file not found: {label_path}")
        return boxes, labels
    
    # Parse each line in the label file
    with open(label_path, "r") as f:
        for line in f.readlines():
            parts = line.strip().split()
            # Valid YOLO line has exactly 5 values
            if len(parts) == 5:
                cls, x, y, w, h = parts
                boxes.append([float(x), float(y), float(w), float(h)])
                labels.append(int(cls))
    
    return boxes, labels


def write_yolo_labels(label_path, boxes, labels):
    """
    Write bounding boxes and labels to a YOLO format label file.
    
    Args:
        label_path (str): Output path for the label file
        boxes (list): List of bounding boxes [x_center, y_center, width, height]
        labels (list): List of class IDs corresponding to each box
    """
    with open(label_path, "w") as f:
        for box, cls in zip(boxes, labels):
            # Format: class x_center y_center width height (6 decimal places)
            f.write(f"{cls} " + " ".join(f"{x:.6f}" for x in box) + "\n")

# ============================================================================
# MAIN AUGMENTATION LOOP
# ============================================================================

# ---- Tunable setting (adjust here directly) ----
N_AUGS = 10  # Number of augmented versions to create per original image

# Get all image files from the input directory
img_files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

print(f"Found {len(img_files)} image files: {img_files}")
print(f"Will generate {N_AUGS} augmentations per image")
print("-" * 60)

# Process each image in the dataset
for img_file in img_files:
    print(f"Processing: {img_file}")
    
    # Construct paths for image and corresponding label file
    img_path = os.path.join(IMG_DIR, img_file)
    lbl_path = os.path.join(LBL_DIR, os.path.splitext(img_file)[0] + ".txt")

    # Load the image using OpenCV
    image = cv2.imread(img_path)
    if image is None:
        print(f"Failed to load image: {img_path}")
        continue
    
    # Get image dimensions
    h, w = image.shape[:2]
    
    # Read the corresponding YOLO label file
    boxes, labels = read_yolo_labels(lbl_path)
    
    print(f"Image shape: {h}x{w}, Found {len(boxes)} bounding boxes")

    # Generate N_AUGS augmented versions of this image
    for i in range(N_AUGS):
        try:
            # Apply augmentation pipeline to image and bounding boxes
            transformed = transform(image=image, bboxes=boxes, class_labels=labels)

            # Extract augmented data
            aug_img = transformed["image"]
            aug_boxes = transformed["bboxes"]
            aug_labels = transformed["class_labels"]

            # Skip if bounding boxes were lost during transformation
            # (can happen if objects are rotated out of frame)
            if not aug_boxes:
                print(f"Skipping aug {i} - no boxes remain after transformation")
                continue

            # Create output filenames with augmentation index
            out_img_file = os.path.splitext(img_file)[0] + f"_aug{i}.jpg"
            out_lbl_file = os.path.splitext(img_file)[0] + f"_aug{i}.txt"

            # Save augmented image
            success = cv2.imwrite(os.path.join(OUT_IMG_DIR, out_img_file), aug_img)
            
            if success:
                # Save corresponding augmented labels
                write_yolo_labels(os.path.join(OUT_LBL_DIR, out_lbl_file), aug_boxes, aug_labels)
                print(f"Created: {out_img_file}")
            else:
                print(f"Failed to write: {out_img_file}")
                
        except Exception as e:
            # Catch and log any errors during augmentation
            print(f"Error during augmentation {i}: {e}")

# ============================================================================
# COMPLETION
# ============================================================================

print("-" * 60)
print("Augmentation finished!")
print(f"Augmented images saved to: {OUT_IMG_DIR}")
print(f"Augmented labels saved to: {OUT_LBL_DIR}")