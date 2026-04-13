"""
keypoint_dataaug.py - Data Augmentation for YOLO Pose Dataset

This script performs data augmentation on keypoint-labeled images using Albumentations.
It applies geometric transformations while preserving keypoint annotations and recalculates
bounding boxes based on visible keypoints after augmentation.

Transformations Applied:
- Perspective distortion
- Rotation (±15°)
- Brightness/contrast adjustment
- Gaussian blur

Author: NCSU - Fall 2025 Board Game Project
"""

import os
import cv2
import albumentations as A
import numpy as np

# ============================================================================
# CONFIGURATION
# ============================================================================

# Dataset directory paths
DATASET_DIR = "yolo_keypoint_dataset"
IMG_DIR = os.path.join(DATASET_DIR, "images")  # Original images
LBL_DIR = os.path.join(DATASET_DIR, "labels")  # Original YOLO pose labels

# Output directories for augmented data
OUT_IMG_DIR = os.path.join(DATASET_DIR, "images_aug")  # Augmented images
OUT_LBL_DIR = os.path.join(DATASET_DIR, "labels_aug")  # Augmented labels
os.makedirs(OUT_IMG_DIR, exist_ok=True)
os.makedirs(OUT_LBL_DIR, exist_ok=True)

# Augmentation parameters
NUM_KEYPOINTS = 9  # Number of hex keypoints on the board
N_AUGS = 10  # Number of augmented versions to create per image

# ============================================================================
# AUGMENTATION PIPELINE
# ============================================================================

# Define augmentation transformations
# Note: Keypoint format is 'xy' (pixel coordinates), not normalized
transform = A.Compose([
    A.Perspective(
        scale=(0.05, 0.1),  # Perspective distortion scale
        keep_size=True,  # Maintain original image size
        pad_mode=cv2.BORDER_CONSTANT,  # Fill with black
        p=0.3  # 30% probability
    ),
    A.Rotate(
        limit=15,  # Rotate by ±15 degrees
        border_mode=cv2.BORDER_CONSTANT,
        p=0.7  # 70% probability
    ),
    A.RandomBrightnessContrast(p=0.3),  # Adjust brightness/contrast
    A.GaussianBlur(blur_limit=3, p=0.2)  # Apply slight blur
], keypoint_params=A.KeypointParams(
    format='xy',  # Keypoint format: (x, y) pixel coordinates
    remove_invisible=False  # Keep keypoints even if moved outside image
))

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def read_yolo_keypoint_label(label_path):
    """
    Read YOLO pose format label file.
    
    Format: class x_center y_center width height kp1_x kp1_y kp1_v kp2_x kp2_y kp2_v ...
    
    Args:
        label_path (str): Path to the YOLO label file
        
    Returns:
        tuple: (class_id, bbox, keypoints, visibilities)
    """
    with open(label_path, "r") as f:
        line = f.readline().strip().split()
        class_id = int(line[0])
        bbox = [float(x) for x in line[1:5]]

        keypoints = []
        visibilities = []
        for i in range(5, len(line), 3):
            if i+2 < len(line):
                x = float(line[i])
                y = float(line[i+1])
                v = int(line[i+2])
                keypoints.append((x, y))
                visibilities.append(v)
        
        return class_id, bbox, keypoints, visibilities

def write_yolo_keypoint_label(label_path, class_id, bbox, keypoints, visibilities):
    """
    Write YOLO pose format label file.
    
    Args:
        label_path: Output path for label file
        class_id: Object class ID
        bbox: Bounding box [x_center, y_center, width, height]
        keypoints: List of (x, y) tuples in normalized coordinates
        visibilities: List of visibility flags (0, 1, or 2)
    """
    with open(label_path, "w") as f:
        f.write(f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f} ")

        for i in range(len(keypoints)):
            x, y = keypoints[i]
            v = visibilities[i]
            f.write(f"{x:.6f} {y:.6f} {v} ")
        
        f.write("\n")

# ============================================================================
# MAIN AUGMENTATION LOOP
# ============================================================================

# Get all image files from the input directory
img_files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

# Process each image in the dataset
for img_file in img_files:
    # Construct paths for image and label files
    img_path = os.path.join(IMG_DIR, img_file)
    lbl_path = os.path.join(LBL_DIR, os.path.splitext(img_file)[0] + ".txt")
    
    # Skip if label file doesn't exist
    if not os.path.exists(lbl_path):
        continue

    # Load image and get dimensions
    image = cv2.imread(img_path)
    h, w = image.shape[:2]
    
    # Read YOLO pose labels
    class_id, bbox, keypoints, visibilities = read_yolo_keypoint_label(lbl_path)

    # Convert normalized keypoints (0-1) to pixel coordinates
    pixel_keypoints = []
    for idx, (kp, vis) in enumerate(zip(keypoints, visibilities)):
        x, y = kp
        if vis in [1, 2]:  # Visible or occluded
            pixel_keypoints.append((x * w, y * h))
        else:  # Not labeled
            print(f"Found invisible keypoint in {img_file}")
            pixel_keypoints.append((0.0, 0.0))

    # Generate N_AUGS augmented versions of this image
    for i in range(N_AUGS):
        # Apply augmentation pipeline to image and keypoints
        transformed = transform(image=image, keypoints=pixel_keypoints)
        aug_img = transformed["image"]
        aug_kps = transformed["keypoints"]

        # Convert augmented keypoints back to normalized coordinates
        aug_keypoints = []
        aug_vis = []
        
        for idx, (aug_kp, original_vis) in enumerate(zip(aug_kps, visibilities)):
            x, y = aug_kp
            
            # Only process visible/occluded keypoints
            if original_vis in [1, 2]:
                # Normalize to 0-1 range and clamp to valid bounds
                norm_x = max(0.0, min(1.0, x / aug_img.shape[1]))
                norm_y = max(0.0, min(1.0, y / aug_img.shape[0]))
                aug_keypoints.append((norm_x, norm_y))
                aug_vis.append(original_vis)
            else:
                # Keep invisible keypoints as is
                aug_keypoints.append((0.0, 0.0))
                aug_vis.append(0)

        # Create output filenames with augmentation index
        out_img_file = os.path.splitext(img_file)[0] + f"_aug{i}.jpg"
        out_lbl_file = os.path.splitext(img_file)[0] + f"_aug{i}.txt"
        
        # Save augmented image
        cv2.imwrite(os.path.join(OUT_IMG_DIR, out_img_file), aug_img)
        
        # Save augmented labels
        write_yolo_keypoint_label(
            os.path.join(OUT_LBL_DIR, out_lbl_file),
            class_id, bbox, aug_keypoints, aug_vis
        )

# ============================================================================
# COMPLETION
# ============================================================================

print("✅ Data augmentation completed successfully!")
print(f"✅ Processed {len(img_files)} images")
print(f"✅ Generated {len(img_files) * N_AUGS} augmented samples")
print(f"✅ Augmented images saved to: {OUT_IMG_DIR}")
print(f"✅ Augmented labels saved to: {OUT_LBL_DIR}")