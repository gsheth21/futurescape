"""
convert_to_keypoint.py - Label Studio JSON to YOLO Pose Format Converter

This script converts keypoint annotations from Label Studio JSON format to YOLO pose format.
YOLO pose format: class x_center y_center width height kp1_x kp1_y kp1_v kp2_x kp2_y kp2_v ...

Where:
- class: object class ID (0 for board)
- x_center, y_center, width, height: bounding box in normalized coordinates (0-1)
- kp_x, kp_y: keypoint coordinates in normalized format (0-1)
- kp_v: visibility flag (0=not labeled, 1=occluded, 2=visible)

Author: NCSU - Fall 2025 Board Game Project
"""

import json
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths to input and output files
input_json = "yolo_keypoint_dataset/yolo_keypoint_dataset.json"  # Label Studio export
output_dir = "yolo_keypoint_dataset/labels"  # YOLO label files output
os.makedirs(output_dir, exist_ok=True)

# Keypoint configuration - 9 hex centers on the board
# These must match the labels used in Label Studio
KEYPOINTS = [f"Hex{i+1}" for i in range(9)]

with open(input_json) as f:
    data = json.load(f)

for item in data:
    # Extract image filename without extension
    image_name = item["file_upload"].split(".")[0]
    
    # Get all annotations for this image
    annos = item["annotations"][0]["result"]

    # Dictionary to store keypoint data: {label: (x, y, visibility)}
    kp_dict = {}
    
    # Parse each annotation
    for kp in annos:
        # Only process keypoint annotations (not bounding boxes, etc.)
        if kp["type"] == "keypointlabels":
            # Get keypoint label (e.g., "Hex1", "Hex2", ...)
            label = kp["value"]["keypointlabels"][0]
            
            # Convert from percentage (0-100) to normalized (0-1)
            x = kp["value"]["x"] / 100
            y = kp["value"]["y"] / 100

            # Determine visibility flag
            # 0 = not labeled, 1 = occluded, 2 = visible
            viz = 2  # Default: visible
            if "meta" in kp:
                if kp["meta"]["text"][0] == "occluded":
                    viz = 1  # Mark as occluded

            # Store keypoint data
            kp_dict[label] = (x, y, viz)
    
    # ========================================================================
    # BUILD KEYPOINT ARRAY IN CORRECT ORDER
    # ========================================================================
    
    keypoint_data = []

    # Ensure keypoints are in the correct order (Hex1, Hex2, ..., Hex9)
    for label in KEYPOINTS:
        if label in kp_dict:
            x, y, viz = kp_dict[label]
            keypoint_data.extend([x, y, viz])
        else:
            # If keypoint is missing, use placeholder values
            print(f"⚠️ Label not found: {label} in file {image_name}")
            keypoint_data.extend([0.0, 0.0, 0])  # Not visible

    # ========================================================================
    # CREATE BOUNDING BOX
    # ========================================================================
    
    # Use full image as bounding box (will be updated during augmentation)
    # Format: x_center, y_center, width, height (all normalized 0-1)
    x_center, y_center, w, h = 0.5, 0.5, 1.0, 1.0

    # ========================================================================
    # WRITE YOLO POSE FORMAT LABEL FILE
    # ========================================================================
    
    txt_path = os.path.join(output_dir, f"{image_name}.txt")
    with open(txt_path, "w") as f_out:
        # Write class ID and bounding box
        f_out.write(f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f} ")

        # Write all keypoints (x, y, visibility) in order
        for i in range(0, len(keypoint_data), 3):
            x = keypoint_data[i]
            y = keypoint_data[i+1]
            v = keypoint_data[i+2]
            f_out.write(f"{x:.6f} {y:.6f} {v} ")
        
        f_out.write("\n")

# ============================================================================
# COMPLETION
# ============================================================================

print("✅ Conversion completed with CORRECT YOLO pose format!")
print(f"✅ Processed {len(data)} images")
print(f"✅ Labels saved to: {output_dir}")