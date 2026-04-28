"""
polyseg_datasug.py - Polygon Segmentation Data Augmentation

This script performs data augmentation on polygon-labeled images for semantic segmentation.
It applies geometric and photometric transformations while preserving polygon annotations.

Transformations Applied:
- Perspective distortion
- Rotation (±15°)
- Brightness/contrast adjustment
- Gaussian blur

The script handles polygon coordinates carefully to maintain annotation integrity.

Author: NCSU - Fall 2025 Board Game Project
"""

import os
import cv2
import numpy as np
import random
from pathlib import Path
from PIL import Image, ImageDraw
import albumentations as A
from dotenv import load_dotenv

# Load .env from board_segmentation/ (one level up from src/)
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

# ============================================================================
# CONFIGURATION
# ============================================================================

# N_AUGS: Tunable setting (adjust here directly)
N_AUGS = 10

# Dataset directory paths (resolved from .env)
DATASET_DIR = Path(os.environ["AUG_DATASET_DIR"])
IMG_DIR = DATASET_DIR / "images"   # Original images
LBL_DIR = DATASET_DIR / "labels"   # Polygon labels in YOLO format

# Output directories for augmented data
OUT_IMG_DIR = DATASET_DIR / "images_aug"  # Augmented images
OUT_LBL_DIR = DATASET_DIR / "labels_aug"  # Augmented polygon labels
OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
OUT_LBL_DIR.mkdir(parents=True, exist_ok=True)

print(f"Looking for images in: {IMG_DIR}")
print(f"Directory exists: {os.path.exists(IMG_DIR)}")

# ============================================================================
# AUGMENTATION PIPELINE
# ============================================================================

# Define augmentation transformations using Albumentations
# Polygons are treated as keypoints during transformation
transform = A.Compose([
    # Note: Flips are commented out - uncomment if needed
    # A.HorizontalFlip(p=0.5),
    # A.VerticalFlip(p=0.5),
    A.Perspective(
        scale=(0.05, 0.1),  # Perspective distortion strength
        keep_size=True,  # Maintain original image size
        pad_mode=cv2.BORDER_CONSTANT,  # Fill with black
        p=0.3  # 30% probability
    ),
    A.Rotate(
        limit=15,  # Rotate by ±15 degrees
        border_mode=cv2.BORDER_CONSTANT,
        p=0.7  # 70% probability
    ),
    A.RandomBrightnessContrast(p=0.3),  # Adjust lighting
    A.GaussianBlur(blur_limit=3, p=0.2)  # Slight blur
], keypoint_params=A.KeypointParams(
    format='xy',  # Keypoint format: (x, y) pixel coordinates
    remove_invisible=True  # Remove keypoints that move outside image
))

def read_polygon_labels(label_path):
    """
    Read polygon labels from YOLO segmentation format text file.
    
    Format: class_id x1 y1 x2 y2 x3 y3 ... xn yn
    - Coordinates are normalized (0-1 range)
    - Minimum 3 points (6 values) required for a valid polygon
    
    Args:
        label_path: Path to label .txt file
        
    Returns:
        polygons: List of polygons, each as [(x1,y1), (x2,y2), ...]
        classes: List of class IDs corresponding to each polygon
    """
    polygons, classes = [], []
    
    # Check if label file exists
    if not os.path.exists(label_path):
        print(f"Label file not found: {label_path}")
        return polygons, classes
    
    # Read each line as a separate polygon
    with open(label_path, "r") as f:
        for line in f.readlines():
            parts = line.strip().split()
            
            # Need at least class + 3 points (1 + 6 values = 7 minimum)
            if len(parts) >= 6:
                cls = int(parts[0])  # First value is class ID
                coords = [float(x) for x in parts[1:]]  # Rest are coordinates
                
                # Ensure even number of coordinates (x,y pairs)
                if len(coords) % 2 == 0:
                    # Convert flat list to list of (x,y) tuples
                    polygon = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]
                    polygons.append(polygon)
                    classes.append(cls)
    
    return polygons, classes

def write_polygon_labels(label_path, polygons, classes):
    """
    Write polygon labels to YOLO segmentation format text file.
    
    Format: class_id x1 y1 x2 y2 x3 y3 ... xn yn
    - Each line represents one polygon
    - Coordinates are normalized to 6 decimal places
    
    Args:
        label_path: Output path for label .txt file
        polygons: List of polygons, each as [(x1,y1), (x2,y2), ...]
        classes: List of class IDs corresponding to each polygon
    """
    with open(label_path, "w") as f:
        for polygon, cls in zip(polygons, classes):
            # Convert polygon points to space-separated string
            coords_str = " ".join([f"{x:.6f} {y:.6f}" for x, y in polygon])
            # Write: class_id x1 y1 x2 y2 ...
            f.write(f"{cls} {coords_str}\n")

def polygon_to_keypoints(polygons):
    """
    Convert polygon list to flat keypoints list for Albumentations transformations.
    
    Albumentations treats polygons as keypoints during augmentation. We need to:
    1. Flatten all polygon points into a single keypoint list
    2. Track how many points belong to each polygon
    
    Args:
        polygons: List of polygons, each as [(x1,y1), (x2,y2), ...]
        
    Returns:
        keypoints: Flat list of all points as (x, y) tuples
        polygon_lengths: List tracking number of points in each polygon
                         Example: [4, 5, 3] means first polygon has 4 points,
                         second has 5, third has 3
    """
    keypoints = []
    polygon_lengths = []
    
    for i, polygon in enumerate(polygons):
        # Record how many points this polygon has
        polygon_lengths.append(len(polygon))
        
        # Add all polygon points to keypoint list
        for j, (x, y) in enumerate(polygon):
            if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                keypoints.append((float(x), float(y)))
    
    return keypoints, polygon_lengths

def keypoints_to_polygons(keypoints, polygon_lengths):
    """
    Convert flat keypoints list back to polygon format after augmentation.
    
    This reverses the polygon_to_keypoints operation by:
    1. Using polygon_lengths to split keypoints back into separate polygons
    2. Filtering out invalid polygons (< 3 points)
    
    Args:
        keypoints: Flat list of transformed keypoints as (x, y) tuples
        polygon_lengths: Original polygon point counts [4, 5, 3, ...]
        
    Returns:
        polygons: List of reconstructed polygons, each as [(x1,y1), (x2,y2), ...]
                  Only includes polygons with >= 3 points
    """
    polygons = []
    start_idx = 0
    
    # Reconstruct each polygon using recorded lengths
    for i, length in enumerate(polygon_lengths):
        end_idx = start_idx + length
        
        # Ensure we don't exceed keypoint list bounds
        if end_idx <= len(keypoints):
            polygon_keypoints = keypoints[start_idx:end_idx]
            polygon = []
            
            # Extract x,y coordinates from each keypoint
            for kp in polygon_keypoints:
                if len(kp) >= 2:
                    polygon.append((kp[0], kp[1]))
            
            # Only keep valid polygons (minimum 3 points)
            if len(polygon) >= 3:
                polygons.append(polygon)
                print(f"Polygon {i}: {len(polygon)} points")
            
            start_idx = end_idx
        else:
            # Ran out of keypoints (some were lost during augmentation)
            break
    
    return polygons

def denormalize_coordinates(polygons, img_width, img_height):
    """
    Convert normalized coordinates (0-1) to pixel coordinates for augmentation.
    
    YOLO format uses normalized coordinates, but Albumentations needs pixel coordinates.
    
    Process:
    1. Clamp coordinates to [0, 1] range (safety check)
    2. Multiply by (width-1, height-1) to get pixel positions
       - Using (width-1) because pixels are 0-indexed
    
    Args:
        polygons: List of polygons with normalized coords [(x1,y1), ...]
        img_width: Image width in pixels
        img_height: Image height in pixels
        
    Returns:
        denorm_polygons: List of polygons with pixel coordinates
    """
    denorm_polygons = []
    
    for polygon in polygons:
        denorm_polygon = []
        for x, y in polygon:
            # Clamp to valid range [0, 1]
            x_clamped = max(0.0, min(1.0, x))
            y_clamped = max(0.0, min(1.0, y))
            
            # Convert to pixel coordinates
            # Use (width-1) because pixels are indexed 0 to (width-1)
            pixel_x = x_clamped * (img_width - 1)
            pixel_y = y_clamped * (img_height - 1)
            
            denorm_polygon.append((pixel_x, pixel_y))
        denorm_polygons.append(denorm_polygon)
    
    return denorm_polygons

def normalize_coordinates(polygons, img_width, img_height):
    """
    Convert pixel coordinates back to normalized coordinates (0-1) after augmentation.
    
    This reverses denormalize_coordinates to save in YOLO format.
    
    Process:
    1. Divide pixel coordinates by (width-1, height-1)
    2. Clamp result to [0, 1] range (safety check)
    3. Handle edge case where width or height is 1
    
    Args:
        polygons: List of polygons with pixel coords [(x1,y1), ...]
        img_width: Image width in pixels
        img_height: Image height in pixels
        
    Returns:
        norm_polygons: List of polygons with normalized coordinates
    """
    norm_polygons = []
    
    for polygon in polygons:
        norm_polygon = []
        for x, y in polygon:
            # Normalize back to [0, 1] range
            # Handle edge case where width/height is 1
            norm_x = x / max(1, img_width - 1) if img_width > 1 else 0
            norm_y = y / max(1, img_height - 1) if img_height > 1 else 0
            
            # Clamp to valid range [0, 1]
            norm_x = max(0.0, min(1.0, norm_x))
            norm_y = max(0.0, min(1.0, norm_y))
            
            norm_polygon.append((norm_x, norm_y))
        norm_polygons.append(norm_polygon)
    
    return norm_polygons

def validate_polygon(polygon, tolerance=0.1):
    """
    Check if polygon is valid after transformation.
    
    Validation checks:
    1. Minimum 3 points required for a polygon
    2. All coordinates must be in valid range [0, 1]
    3. Polygon must have non-zero area (not collapsed to a line)
    
    Args:
        polygon: List of (x, y) tuples in normalized coordinates
        tolerance: Unused parameter (kept for compatibility)
        
    Returns:
        bool: True if polygon is valid, False otherwise
    """
    # Check minimum point count
    if len(polygon) < 3:
        return False
    
    # Check if all coordinates are within bounds [0, 1]
    for x, y in polygon:
        if not (0 <= x <= 1 and 0 <= y <= 1):
            return False
    
    # Calculate polygon area using Shoelace formula
    # Area = |Σ(x_i * y_(i+1) - x_(i+1) * y_i)| / 2
    area = 0
    n = len(polygon)
    for i in range(n):
        j = (i + 1) % n  # Next vertex (wraps around)
        area += polygon[i][0] * polygon[j][1]
        area -= polygon[j][0] * polygon[i][1]
    area = abs(area) / 2.0
    
    # Polygon is valid if area > 0 (not collapsed to a line/point)
    return area > 1e-6 

# ============================================================================
# MAIN AUGMENTATION LOOP
# ============================================================================

# Get all image files from input directory
img_files = [f.name for f in IMG_DIR.iterdir() if f.suffix.lower() in (".jpg", ".png", ".jpeg")]

print(f"Found {len(img_files)} image files: {img_files}")

# Track success rate
success_count = 0
total_attempts = 0

# Process each image in the dataset
for img_file in img_files:
    print(f"\n{'='*60}")
    print(f"Processing: {img_file}")
    print(f"{'='*60}")
    
    # Construct file paths
    img_path = IMG_DIR / img_file
    lbl_path = LBL_DIR / (os.path.splitext(img_file)[0] + ".txt")

    # Load image using OpenCV
    image = cv2.imread(img_path)
    if image is None:
        print(f"❌ Failed to load image: {img_path}")
        continue
        
    h, w = image.shape[:2]
    
    # Load corresponding polygon labels
    polygons, classes = read_polygon_labels(lbl_path)
    
    print(f"📐 Image shape: {h}x{w}")
    print(f"🔷 Found {len(polygons)} polygons with classes: {classes}")
    
    # Skip images without polygon annotations
    if not polygons:
        print(f"⚠️  No polygons found for {img_file}, skipping...")
        continue

    # ========================================================================
    # SAVE ORIGINAL IMAGE AND LABELS
    # ========================================================================
    # First, copy original files to output directory with "_orig" suffix
    # This ensures we keep unmodified versions in the augmented dataset
    
    orig_img_file = os.path.splitext(img_file)[0] + "_orig.jpg"
    orig_lbl_file = os.path.splitext(img_file)[0] + "_orig.txt"
    
    cv2.imwrite(str(OUT_IMG_DIR / orig_img_file), image)
    write_polygon_labels(str(OUT_LBL_DIR / orig_lbl_file), polygons, classes)
    success_count += 1
    print(f"✅ Copied original: {orig_img_file}")

    # ========================================================================
    # PREPARE FOR AUGMENTATION
    # ========================================================================
    # Convert normalized coordinates (0-1) to pixel coordinates
    # Albumentations works with pixel coordinates, not normalized ones
    pixel_polygons = denormalize_coordinates(polygons, w, h)
    
    # ========================================================================
    # GENERATE AUGMENTED VERSIONS
    # ========================================================================
    for i in range(N_AUGS):
        total_attempts += 1
        
        try:
            # ----------------------------------------------------------------
            # Step 1: Convert polygons to keypoints format
            # ----------------------------------------------------------------
            # Albumentations treats polygon vertices as keypoints during transformation
            keypoints, polygon_lengths = polygon_to_keypoints(pixel_polygons)
            
            # ----------------------------------------------------------------
            # Step 2: Apply augmentation transformations
            # ----------------------------------------------------------------
            # This applies: perspective distortion, rotation, brightness, blur
            # Keypoints are transformed along with the image
            transformed = transform(image=image, keypoints=keypoints)
            
            aug_img = transformed["image"]
            aug_keypoints = transformed["keypoints"]

            # ----------------------------------------------------------------
            # Step 3: Check if keypoints were lost during transformation
            # ----------------------------------------------------------------
            # Some keypoints may move outside image bounds and get removed
            if len(aug_keypoints) != len(keypoints):
                print(f"⚠️  Aug {i}: Lost keypoints during transformation "
                      f"({len(keypoints)} -> {len(aug_keypoints)})")
                continue
            
            # ----------------------------------------------------------------
            # Step 4: Reconstruct polygons from transformed keypoints
            # ----------------------------------------------------------------
            aug_polygons = keypoints_to_polygons(aug_keypoints, polygon_lengths)
            
            # ----------------------------------------------------------------
            # Step 5: Normalize coordinates back to [0, 1] range
            # ----------------------------------------------------------------
            aug_h, aug_w = aug_img.shape[:2]
            norm_aug_polygons = normalize_coordinates(aug_polygons, aug_w, aug_h)
            
            # ----------------------------------------------------------------
            # Step 6: Validate and filter polygons
            # ----------------------------------------------------------------
            # Some polygons may become invalid after transformation:
            # - Points moved outside image bounds
            # - Polygon collapsed to a line (zero area)
            # - Less than 3 points remaining
            
            valid_polygons = []
            valid_classes = []
            
            for polygon, cls in zip(norm_aug_polygons, classes):
                if validate_polygon(polygon):
                    valid_polygons.append(polygon)
                    valid_classes.append(cls)
            
            # Skip this augmentation if no valid polygons remain
            if not valid_polygons:
                print(f"❌ Skipping aug {i} - no valid polygons remain after transformation")
                continue

            # ----------------------------------------------------------------
            # Step 7: Save augmented image and labels
            # ----------------------------------------------------------------
            out_img_file = os.path.splitext(img_file)[0] + f"_aug{i}.jpg"
            out_lbl_file = os.path.splitext(img_file)[0] + f"_aug{i}.txt"

            # Write augmented image
            success = cv2.imwrite(str(OUT_IMG_DIR / out_img_file), aug_img)

            if success:
                # Write augmented polygon labels
                write_polygon_labels(str(OUT_LBL_DIR / out_lbl_file), 
                                     valid_polygons, valid_classes)
                print(f"✅ Created: {out_img_file} with {len(valid_polygons)} polygons")
                success_count += 1
            else:
                print(f"❌ Failed to write: {out_img_file}")
                
        except Exception as e:
            print(f"❌ Error during augmentation {i}: {e}")

# ============================================================================
# SUMMARY
# ============================================================================
print(f"\n{'='*60}")
print(f"Polygon augmentation finished!")
print(f"{'='*60}")
print(f"✅ Successfully created: {success_count} images")
print(f"❌ Failed attempts: {total_attempts - success_count}")
print(f"📊 Success rate: {success_count/total_attempts*100:.1f}%")
print(f"📁 Output directories:")
print(f"   Images: {OUT_IMG_DIR}")
print(f"   Labels: {OUT_LBL_DIR}")
print(f"{'='*60}")