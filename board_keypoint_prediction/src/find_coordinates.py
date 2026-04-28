"""
find_coordinates.py - Interactive Hex Center Coordinate Finder

This script allows you to manually click on hex centers in an image to extract their pixel coordinates.
Useful for creating initial keypoint annotations or verifying predicted keypoint locations.

Usage:
    1. Update the image path below
    2. Run the script
    3. Click on each hex center in the image
    4. Close the window when done
    5. The coordinates will be printed to console

Author: NCSU - Fall 2025 Board Game Project
"""

import os
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from dotenv import load_dotenv

# ============================================================================
# CONFIGURATION
# ============================================================================

# Load .env from the board_keypoint_prediction/ directory (one level up from src/)
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

# Load your image (path set via .env)
img = cv2.imread(os.environ["IDEAL_IMAGE_PATH"])
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# List to store clicked coordinates
clicked_points = []

# ============================================================================
# INTERACTIVE CLICKING FUNCTION
# ============================================================================

def onclick(event):
    """
    Callback function for mouse click events.
    Stores clicked coordinates and displays them on the image.
    
    Args:
        event: matplotlib mouse click event
    """
    if event.xdata is not None and event.ydata is not None:
        x, y = int(event.xdata), int(event.ydata)
        clicked_points.append((x, y))
        print(f"Clicked at: ({x}, {y})")
        
        # Draw red dot at clicked location
        plt.scatter([x], [y], c='red', s=100, marker='o')
        plt.draw()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

# Create figure and display image
fig, ax = plt.subplots(figsize=(12, 8))
ax.imshow(img_rgb)
ax.set_title("Click on hex centers (close window when done)", fontsize=14)

# Connect click event handler
cid = fig.canvas.mpl_connect('button_press_event', onclick)

# Show interactive window
plt.show()

# ============================================================================
# OUTPUT RESULTS
# ============================================================================

print("\n" + "="*60)
print("CLICKED COORDINATES")
print("="*60)
print(f"Total points clicked: {len(clicked_points)}")
print("\nAll clicked points:")
for i, (x, y) in enumerate(clicked_points, 1):
    print(f"  Hex{i}: ({x}, {y})")