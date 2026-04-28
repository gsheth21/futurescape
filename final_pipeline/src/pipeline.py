"""
pipeline.py  —  End-to-end: image(s) → hexid-piece mapping

Given a raw test image (or directory of images), this script:
  1. Detects & crops the board                   (YOLO board detector)
  2. Runs SuperPoint on the crop                 (feature extraction + matching)
  3. Projects hex centers into crop space        (homography + projection)
  4. Detects red pieces in the crop              (YOLO piece detector)
  5. Assigns pieces to hex cells                 (containment + distance fallback)
  6. Saves <name>_hex_piece_map.json + viz

Usage:
    python pipeline.py --image path/to/image.jpg --output_dir path/to/out/
    python pipeline.py --image_dir path/to/images/ --output_dir path/to/out/
"""

import os
import cv2
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

# ── LOAD CONFIG FROM .env ─────────────────────────────────────────────────────

load_dotenv(Path(__file__).resolve().parent.parent / '.env')

SRC_DIR             = Path(os.environ['SUPERPOINT_SRC_DIR'])

BOARD_YOLO_WEIGHTS  = os.environ['BOARD_YOLO_WEIGHTS']
PIECE_YOLO_WEIGHTS  = os.environ['PIECE_YOLO_WEIGHTS']
SP_WEIGHTS          = os.environ['SP_WEIGHTS']
TEMPLATE_DIR        = Path(os.environ['TEMPLATE_DIR'])
HEX_CENTERS_PATH    = Path(os.environ['HEX_CENTERS_PATH'])

BOARD_CONF  = float(os.environ.get('BOARD_CONF', 0.5))
PIECE_CONF  = float(os.environ.get('PIECE_CONF', 0.5))
BOARD_PAD   = float(os.environ.get('BOARD_PAD',  0.05))


# ── LAZY IMPORTS (so paths are resolved before importing SP) ──────────────────

def _load_superpoint():
    sys.path.insert(0, str(SRC_DIR))
    from superpoint import load_model, detect
    from matcher import match, compute_homography
    return load_model(SP_WEIGHTS), detect, match, compute_homography


# ── STEP 1: CROP BOARD ────────────────────────────────────────────────────────

def crop_board(image_path, board_model):
    from ultralytics import YOLO

    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    H, W = img.shape[:2]

    results = board_model(img, conf=BOARD_CONF, verbose=False)[0]
    if results.boxes is None or len(results.boxes) == 0:
        return None, None

    best = results.boxes[results.boxes.conf.argmax()]
    x1, y1, x2, y2 = best.xyxy[0].tolist()

    # padding
    pad_x = int((x2 - x1) * BOARD_PAD)
    pad_y = int((y2 - y1) * BOARD_PAD)
    x1 = max(0, int(x1) - pad_x)
    y1 = max(0, int(y1) - pad_y)
    x2 = min(W, int(x2) + pad_x)
    y2 = min(H, int(y2) + pad_y)

    crop = img[y1:y2, x1:x2]
    print(f"  [Crop]  Board bbox: ({x1},{y1},{x2},{y2})  conf={float(best.conf[0]):.2f}")
    return crop, (x1, y1, x2, y2)


# ── STEP 2: SUPERPOINT + HOMOGRAPHY → HEX CENTERS ────────────────────────────

MAX_LONG_SIDE = 960  # must match preprocessing applied to the template


def _resize_for_superpoint(crop_img):
    """
    Resize crop so the long side is at most MAX_LONG_SIDE, matching the
    preprocessing applied to the ideal template before its .npy was generated.
    Returns (resized_img, scale) where scale = resized / original.
    If already small enough, returns (crop_img, 1.0).
    """
    h, w = crop_img.shape[:2]
    long_side = max(h, w)
    if long_side <= MAX_LONG_SIDE:
        return crop_img, 1.0

    scale = MAX_LONG_SIDE / long_side
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(crop_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def project_hex_centers(crop_img, sp_model, detect_fn, match_fn, homography_fn,
                         template_tensor, pts_tmpl, desc_tmpl, hex_centers_template):
    import torch

    # Resize crop to match template preprocessing scale, track scale factor
    crop_resized, scale = _resize_for_superpoint(crop_img)
    print(f"  [SP]    Crop resized: {crop_img.shape[1]}x{crop_img.shape[0]} "
          f"-> {crop_resized.shape[1]}x{crop_resized.shape[0]}  (scale={scale:.3f})")

    gray = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    tensor = torch.tensor(gray).unsqueeze(0).unsqueeze(0)

    pts_new, desc_new, _ = detect_fn(sp_model, tensor)
    matches = match_fn(desc_tmpl, desc_new)

    if len(matches) < 4:
        print(f"  [SP]    Not enough matches ({len(matches)}) — skipping")
        return None

    H, mask = homography_fn(pts_tmpl, pts_new, matches)
    if H is None:
        print("  [SP]    Homography failed — skipping")
        return None

    inliers = int(mask.sum()) if mask is not None else len(matches)
    print(f"  [SP]    Matches: {len(matches)}  Inliers: {inliers}")

    # Project hex centers — these coords are in resized crop space
    hex_ids = list(hex_centers_template.keys())
    pts = np.array([hex_centers_template[k] for k in hex_ids], dtype=np.float32)
    projected = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), H).reshape(-1, 2)

    # Scale back to original crop space (piece detection uses original crop)
    if scale != 1.0:
        projected = projected / scale

    hex_centers_crop = {
        hid: (round(float(x), 2), round(float(y), 2))
        for hid, (x, y) in zip(hex_ids, projected)
    }
    print(f"  [Hex]   Projected {len(hex_centers_crop)} hex centers (in original crop space)")
    return hex_centers_crop


# ── STEP 3: DETECT PIECES ─────────────────────────────────────────────────────

def detect_pieces(crop_img, piece_model):
    results = piece_model(crop_img, conf=PIECE_CONF, verbose=False)[0]
    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append({
            "center": ((x1 + x2) / 2, (y1 + y2) / 2),
            "bbox":   (x1, y1, x2, y2),
            "conf":   float(box.conf[0]),
        })
    print(f"  [Pieces] Detected {len(detections)} pieces")
    return detections


# ── STEP 4: ASSIGN PIECES → HEXES ────────────────────────────────────────────

def estimate_hex_radius(hex_centers):
    pts = np.array(list(hex_centers.values()), dtype=np.float32)
    if len(pts) < 2:
        return 50.0
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1))
    np.fill_diagonal(dist, np.inf)
    radius = float(np.median(dist.min(axis=1))) * 0.6
    print(f"  [Hex]   Radius threshold: {radius:.1f} px")
    return radius


def assign_pieces_to_hexes(hex_centers, detections):
    radius = estimate_hex_radius(hex_centers)
    hex_map      = {hid: 0    for hid in hex_centers}
    matched_dets = {hid: None for hid in hex_centers}

    if not detections:
        return hex_map, matched_dets

    hex_ids = list(hex_centers.keys())
    hex_pts = np.array([hex_centers[hid] for hid in hex_ids], dtype=np.float32)

    for det in detections:
        cx, cy = det["center"]
        dists = np.hypot(hex_pts[:, 0] - cx, hex_pts[:, 1] - cy)

        nearest_idx  = int(np.argmin(dists))
        nearest_dist = dists[nearest_idx]

        if nearest_dist <= 2 * radius:
            hid = hex_ids[nearest_idx]
            hex_map[hid]      = 1
            matched_dets[hid] = det

    occupied = sum(hex_map.values())
    print(f"  [Map]   {occupied} / {len(hex_map)} hexes occupied")
    return hex_map, matched_dets


# ── STEP 5: SAVE OUTPUTS ──────────────────────────────────────────────────────

def save_outputs(name, crop_img, hex_centers, hex_map, detections,
                 matched_dets, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save cropped board image
    crop_path = output_dir / f"{name}_crop.png"
    cv2.imwrite(str(crop_path), crop_img)

    # Save hex-piece map JSON
    json_path = output_dir / f"{name}_hex_piece_map.json"
    with open(json_path, 'w') as f:
        json.dump(hex_map, f, indent=2)
    print(f"  [Save]  JSON  -> {json_path}")

    # Visualization
    canvas = crop_img.copy()

    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 100, 0), 2)
        cv2.putText(canvas, f"{det['conf']:.2f}", (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 100, 0), 1)

    for hid, (hx, hy) in hex_centers.items():
        color = (0, 200, 0) if hex_map[hid] == 1 else (120, 120, 120)
        cv2.circle(canvas, (int(hx), int(hy)), 6, color, -1)
        cv2.circle(canvas, (int(hx), int(hy)), 6, (0, 0, 0), 1)
        cv2.putText(canvas, hid, (int(hx) + 7, int(hy) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, color, 1, cv2.LINE_AA)

    h = canvas.shape[0]
    cv2.putText(canvas,
                "Green=occupied  Grey=empty  Blue box=detected piece",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

    viz_path = output_dir / f"{name}_hex_piece_viz.png"
    cv2.imwrite(str(viz_path), canvas)
    print(f"  [Save]  Viz   -> {viz_path}")


# ── PIPELINE ──────────────────────────────────────────────────────────────────

def run_pipeline(image_path, board_model, piece_model,
                 sp_model, detect_fn, match_fn, homography_fn,
                 template_tensor, pts_tmpl, desc_tmpl, hex_centers_template,
                 output_dir):

    name = Path(image_path).stem
    print(f"\n{'='*60}\nProcessing: {name}")

    # 1. Crop board
    crop_img, bbox = crop_board(image_path, board_model)
    if crop_img is None:
        print("  Board not detected — skipping.")
        return None

    # 2. Project hex centers into crop space
    hex_centers = project_hex_centers(
        crop_img, sp_model, detect_fn, match_fn, homography_fn,
        template_tensor, pts_tmpl, desc_tmpl, hex_centers_template
    )
    if hex_centers is None:
        return None

    # 3. Detect pieces
    detections = detect_pieces(crop_img, piece_model)

    # 4. Assign
    hex_map, matched_dets = assign_pieces_to_hexes(hex_centers, detections)

    # 5. Save
    save_outputs(name, crop_img, hex_centers, hex_map, detections,
                 matched_dets, output_dir)

    return hex_map


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Board game pipeline: image → hex-piece map")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--image',     type=str, help='Path to a single image')
    group.add_argument('--image_dir', type=str, help='Path to a directory of images')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save all outputs')
    args = parser.parse_args()

    from ultralytics import YOLO
    import torch

    print("Loading models...")
    board_model = YOLO(BOARD_YOLO_WEIGHTS)
    piece_model = YOLO(PIECE_YOLO_WEIGHTS)
    sp_model, detect_fn, match_fn, homography_fn = _load_superpoint()

    # Load template once
    from preprocessing import images_to_tensors
    template_tensors = images_to_tensors(str(TEMPLATE_DIR))
    _, template_tensor = next(iter(template_tensors.items()))
    pts_tmpl, desc_tmpl, _ = detect_fn(sp_model, template_tensor)
    print(f"Template: {pts_tmpl.shape[1]} keypoints detected")

    with open(HEX_CENTERS_PATH, 'r') as f:
        hex_centers_template = json.load(f)
    print(f"Hex centers loaded: {len(hex_centers_template)}")

    # Collect images
    if args.image:
        image_paths = [Path(args.image)]
    else:
        image_dir = Path(args.image_dir)
        image_paths = sorted(
            list(image_dir.glob('*.jpg')) +
            list(image_dir.glob('*.jpeg')) +
            list(image_dir.glob('*.png'))
        )
        if not image_paths:
            print(f"No images found in {image_dir}")
            return
        print(f"Found {len(image_paths)} images")

    for image_path in image_paths:
        try:
            run_pipeline(
                image_path, board_model, piece_model,
                sp_model, detect_fn, match_fn, homography_fn,
                template_tensor, pts_tmpl, desc_tmpl, hex_centers_template,
                args.output_dir
            )
        except Exception as e:
            print(f"  ERROR on {image_path.name}: {e}")

    print("\nAll done.")


if __name__ == '__main__':
    main()