import os
import cv2
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from ultralytics import YOLO

load_dotenv(Path(__file__).parents[2] / '.env')

YOLO_WEIGHTS = os.getenv('YOLO_BOARD_WEIGHTS', '')
PADDING      = 0.05   # 5% padding around detected bbox


# ── CORE ─────────────────────────────────────────────────────────────────────

def detect_board(image_path, weights=None, conf_thresh=0.5):
    """
    Run YOLO on an image and return the bounding box of the board.

    Args:
        image_path  : path to image
        weights     : path to trained YOLO weights (defaults to YOLO_BOARD_WEIGHTS env var)
        conf_thresh : minimum confidence to accept detection

    Returns:
        bbox : (x1, y1, x2, y2) in absolute pixel coordinates, or None if not detected
    """
    if weights is None:
        weights = YOLO_WEIGHTS

    model  = YOLO(weights)
    result = model(str(image_path), verbose=False)[0]

    if result.boxes is None or len(result.boxes) == 0:
        print(f"No board detected in: {image_path}")
        return None

    boxes  = result.boxes
    best   = boxes[boxes.conf.argmax()]
    conf   = float(best.conf[0])

    if conf < conf_thresh:
        print(f"Detection confidence too low: {conf:.2f} < {conf_thresh}")
        return None

    x1, y1, x2, y2 = best.xyxy[0].tolist()
    print(f"Board detected | conf: {conf:.2f} | bbox: ({int(x1)}, {int(y1)}, {int(x2)}, {int(y2)})")

    return (int(x1), int(y1), int(x2), int(y2))


def add_padding(bbox, image_shape, padding=PADDING):
    """
    Expand bbox by a percentage of its size, clamped to image boundaries.

    Args:
        bbox        : (x1, y1, x2, y2)
        image_shape : (H, W) of the image
        padding     : fraction to expand (0.05 = 5%)

    Returns:
        padded bbox : (x1, y1, x2, y2)
    """
    H, W    = image_shape
    x1, y1, x2, y2 = bbox

    pad_x = int((x2 - x1) * padding)
    pad_y = int((y2 - y1) * padding)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(W, x2 + pad_x)
    y2 = min(H, y2 + pad_y)

    return (x1, y1, x2, y2)


def crop_board(image_path, weights=None, padding=PADDING):
    """
    Detect board, apply padding, and return the cropped image + bbox used.

    Args:
        image_path : path to image
        weights    : path to trained YOLO weights (defaults to YOLO_BOARD_WEIGHTS env var)
        padding    : padding fraction around bbox

    Returns:
        crop : cropped numpy image (H, W, 3), or None if detection failed
        bbox : (x1, y1, x2, y2) padded bbox used for crop, or None
    """
    if weights is None:
        weights = YOLO_WEIGHTS

    img  = cv2.imread(str(image_path))
    bbox = detect_board(image_path, weights)

    if bbox is None:
        return None, None

    x1, y1, x2, y2 = add_padding(bbox, img.shape[:2], padding)
    crop = img[y1:y2, x1:x2]

    return crop, (x1, y1, x2, y2)


# ── VISUAL CHECK ─────────────────────────────────────────────────────────────

def visualize_detection(image_path, weights=None, save_path=None):
    """
    Draw the detected bounding box on the image to visually verify detection.

    Args:
        image_path : path to image
        weights    : path to trained YOLO weights (defaults to YOLO_BOARD_WEIGHTS env var)
        save_path  : optional path to save the visualization
    """
    if weights is None:
        weights = YOLO_WEIGHTS

    img  = cv2.imread(str(image_path))
    bbox = detect_board(image_path, weights)

    if bbox is None:
        print("Nothing to visualize — no detection.")
        return

    x1, y1, x2, y2 = bbox
    px1, py1, px2, py2 = add_padding(bbox, img.shape[:2])

    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(img, "detected", (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.rectangle(img, (px1, py1), (px2, py2), (0, 255, 0), 2)
    cv2.putText(img, "padded", (px1, py1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    if save_path:
        cv2.imwrite(str(save_path), img)
        print(f"Saved visualization: {save_path}")

    cv2.imshow("Board Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ── QUICK TEST ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='Path to image')
    parser.add_argument('--save',  type=str, default=None,  help='Optional save path for visualization')
    args = parser.parse_args()

    visualize_detection(args.image, save_path=args.save)
