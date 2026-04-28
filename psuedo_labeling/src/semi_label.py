# ============================================================================
# semi_label.py
# Auto-label game pieces in a directory of images using a trained YOLOv8 model.
#
# Two detection modes (set DETECTION_MODE in ../.env):
#   single  — single-class model; all detections are written with CLASS_ID
#   multi   — multi-class model (e.g. all-pieces); class IDs come from the model
#
# Output per image: a .txt file in LABELS_DIR with one line per detection:
#   class_id cx cy w h   (YOLO normalized format)
#
# Configure paths in ../.env before running.
# ============================================================================

import os
from pathlib import Path
from dotenv import load_dotenv
import cv2
from ultralytics import YOLO

# Load .env from the project root (one level above src/)
load_dotenv(Path(__file__).parent.parent / ".env")

IMAGES_DIR       = Path(os.environ["IMAGES_DIR"])
LABELS_DIR       = Path(os.environ["LABELS_DIR"])
BOARD_MODEL_PATH = Path(os.environ["BOARD_MODEL_PATH"])
PIECE_MODEL_PATH = Path(os.environ["PIECE_MODEL_PATH"])

BOARD_CONF       = float(os.getenv("BOARD_CONF", "0.3"))
PIECE_CONF       = float(os.getenv("PIECE_CONF", "0.25"))
CLASS_ID         = int(os.getenv("CLASS_ID", "0"))
DETECTION_MODE   = os.getenv("DETECTION_MODE", "single").strip().lower()

if DETECTION_MODE not in ("single", "multi"):
    raise ValueError(f"DETECTION_MODE must be 'single' or 'multi', got: {DETECTION_MODE!r}")

def run():
    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Detection mode: {DETECTION_MODE}")
    board_model = YOLO(str(BOARD_MODEL_PATH))
    piece_model = YOLO(str(PIECE_MODEL_PATH))

    image_paths = sorted(
        list(IMAGES_DIR.glob("*.jpg")) +
        list(IMAGES_DIR.glob("*.jpeg")) +
        list(IMAGES_DIR.glob("*.png"))
    )

    if not image_paths:
        print(f"No images found in {IMAGES_DIR}")
        return

    print(f"Processing {len(image_paths)} images...")

    skipped = 0
    total_pieces = 0

    for img_path in image_paths:
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  [WARN] Could not read {img_path.name}, skipping.")
            skipped += 1
            continue

        ih, iw = image.shape[:2]

        # Stage 1: detect board and crop to it
        board_results = board_model(image, conf=BOARD_CONF, verbose=False)
        board_boxes   = board_results[0].boxes

        if len(board_boxes) == 0:
            print(f"  [WARN] No board detected in {img_path.name}, skipping.")
            skipped += 1
            continue

        bx1, by1, bx2, by2 = map(int, board_boxes[board_boxes.conf.argmax()].xyxy[0].tolist())
        bx1, by1 = max(0, bx1), max(0, by1)
        bx2, by2 = min(iw, bx2), min(ih, by2)
        crop = image[by1:by2, bx1:bx2]

        # Stage 2: detect yellow pieces in the crop
        piece_results = piece_model(crop, conf=PIECE_CONF, verbose=False)

        lines = []
        for box in piece_results[0].boxes:
            px1, py1, px2, py2 = box.xyxy[0].tolist()
            # Map crop coords back to original image coords
            ox1 = px1 + bx1
            oy1 = py1 + by1
            ox2 = px2 + bx1
            oy2 = py2 + by1
            cx = ((ox1 + ox2) / 2) / iw
            cy = ((oy1 + oy2) / 2) / ih
            w  = (ox2 - ox1) / iw
            h  = (oy2 - oy1) / ih
            # single: use fixed CLASS_ID; multi: use the model's predicted class
            cid = CLASS_ID if DETECTION_MODE == "single" else int(box.cls[0].item())
            lines.append(f"{cid} {cx:.16f} {cy:.16f} {w:.16f} {h:.16f}")

        label_path = LABELS_DIR / (img_path.stem + ".txt")
        label_path.write_text("\n".join(lines))

        total_pieces += len(lines)
        print(f"  {img_path.name}: {len(lines)} pieces → {label_path.name}")

    print(f"\nDone. {len(image_paths) - skipped} images labelled, "
          f"{total_pieces} total pieces, {skipped} skipped.")


if __name__ == "__main__":
    run()