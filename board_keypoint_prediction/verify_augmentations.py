"""
verify_augmentations.py - Augmentation Verification Tool

This script creates preview images showing bounding boxes on augmented samples
to verify that augmentation was applied correctly. Useful for debugging data augmentation.

Note: This script is designed for bounding box visualization.
For keypoint verification, use visualization in the training notebook.

Author: NCSU - Fall 2025 Board Game Project
"""

import os
import cv2
import random

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths to augmented dataset
DATASET_DIR = "Dataset"
IMG_DIR = os.path.join(DATASET_DIR, "images_aug")
LBL_DIR = os.path.join(DATASET_DIR, "labels_aug")
OUT_DIR = os.path.join(DATASET_DIR, "preview")

# Create preview output directory
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def read_yolo_labels(label_path):
    """
    Read YOLO format labels for bounding boxes.
    
    Format: class x_center y_center width height
    
    Args:
        label_path (str): Path to YOLO label file
        
    Returns:
        tuple: (boxes, labels) where boxes are [x, y, w, h] and labels are class IDs
    """
    boxes, labels = [], []
    if not os.path.exists(label_path):
        return boxes, labels
    
    with open(label_path, "r") as f:
        for line in f.readlines():
            parts = line.strip().split()
            # Parse bounding box (first 5 values)
            if len(parts) >= 5:
                cls, x, y, w, h = map(float, parts[:5])
                boxes.append([x, y, w, h])
                labels.append(int(cls))
    
    return boxes, labels

# ============================================================================
# MAIN VERIFICATION LOOP
# ============================================================================

# Get all augmented image files
img_files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

# Randomly sample 10 images for preview (or all if fewer than 10)
sampled_imgs = random.sample(img_files, min(10, len(img_files)))

print(f"Creating previews for {len(sampled_imgs)} random augmented images...")

for img_file in sampled_imgs:
    # Load image and corresponding label
    img_path = os.path.join(IMG_DIR, img_file)
    lbl_path = os.path.join(LBL_DIR, os.path.splitext(img_file)[0] + ".txt")

    # Read image
    image = cv2.imread(img_path)
    h, w = image.shape[:2]

    # Read bounding box labels
    boxes, labels = read_yolo_labels(lbl_path)

    # Draw bounding boxes on the image
    for box, cls in zip(boxes, labels):
        x_center, y_center, bw, bh = box
        
        # Convert YOLO normalized format (0-1) to pixel coordinates
        x1 = int((x_center - bw/2) * w)
        y1 = int((y_center - bh/2) * h)
        x2 = int((x_center + bw/2) * w)
        y2 = int((y_center + bh/2) * h)

        # Draw blue rectangle for bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # Add class label text
        cv2.putText(image, f"Class {cls}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Save preview image with bounding boxes drawn
    out_path = os.path.join(OUT_DIR, f"preview_{img_file}")
    cv2.imwrite(out_path, image)

# ============================================================================
# COMPLETION
# ============================================================================

print(f"✅ Saved {len(sampled_imgs)} preview images with bounding boxes in {OUT_DIR}")
print(f"✅ Review these images to verify augmentation quality")
