import os
import sys
import cv2
import torch
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parents[2] / '.env')

sys.path.insert(0, str(Path(__file__).parent.parent / 'shared'))

from preprocessing import normalize_images, images_to_tensors
from superpoint import load_model, detect
from matcher import match, compute_homography
from predictor import project_keypoints
from visualization import draw_keypoints, draw_matches
from extract import crop_board


# ── CONFIG ───────────────────────────────────────────────────────────────────

SP_WEIGHTS         = os.getenv('SP_WEIGHTS')
YOLO_WEIGHTS       = os.getenv('YOLO_BOARD_WEIGHTS')

TEMPLATE_DIR       = os.getenv('CROPPED_TEMPLATE_DIR')
TEST_IMAGE_DIR     = os.getenv('CROPPED_TEST_IMAGE_DIR')
RESULTS_DIR        = os.getenv('CROPPED_RESULTS_DIR')
CANONICAL_KPS_PATH = os.getenv('CROPPED_CANONICAL_KPS_PATH')


# ── HELPERS ──────────────────────────────────────────────────────────────────

def save_predicted_keypoints(name, projected_kps, results_dir):
    """Save predicted keypoints as .npy for later evaluation."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    save_path = results_dir / f"{name}_pred.npy"
    np.save(save_path, projected_kps)
    print(f"  Saved predicted kps -> {save_path}")


def save_bbox(name, bbox, results_dir):
    """Save bbox (x1, y1, x2, y2) for GT adjustment in evaluator."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    save_path = results_dir / f"{name}_bbox.npy"
    np.save(save_path, np.array(bbox))
    print(f"  Saved bbox -> {save_path}")


def crop_and_tensor(image_path):
    crop, bbox = crop_board(image_path, weights=YOLO_WEIGHTS)

    if crop is None:
        return None, None

    gray   = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    norm   = gray.astype(np.float32) / 255.0
    tensor = torch.tensor(norm).unsqueeze(0).unsqueeze(0)
    return tensor, bbox


# ── RUN ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    fe            = load_model(SP_WEIGHTS)
    canonical_kps = np.load(CANONICAL_KPS_PATH)
    print(f"Loaded {len(canonical_kps)} canonical keypoints")

    template_tensors               = images_to_tensors(TEMPLATE_DIR)
    template_name, template_tensor = next(iter(template_tensors.items()))
    pts_tmpl, desc_tmpl, _         = detect(fe, template_tensor)
    print(f"Template '{template_name}': {pts_tmpl.shape[1]} keypoints detected")

    test_image_paths = sorted(Path(TEST_IMAGE_DIR).glob('*.png'))

    for image_path in test_image_paths:
        name = image_path.stem
        print(f"\nProcessing: {name}")

        tensor, bbox = crop_and_tensor(image_path)

        if tensor is None:
            print(f"  Skipping {name} — board not detected by YOLO")
            continue

        x1, y1, x2, y2 = bbox
        print(f"  Cropped board: bbox=({x1},{y1},{x2},{y2})")

        pts_new, desc_new, _ = detect(fe, tensor)
        matches              = match(desc_tmpl, desc_new)
        H, mask              = compute_homography(pts_tmpl, pts_new, matches)

        if H is None:
            print(f"  Skipping {name} — homography failed")
            continue

        print(f"  Matches: {len(matches)} | Inliers: {int(mask.sum())}")

        projected_kps = project_keypoints(canonical_kps, H)

        save_predicted_keypoints(name, projected_kps, RESULTS_DIR)
        save_bbox(name, bbox, RESULTS_DIR)

        crop_path = Path(RESULTS_DIR) / f"{name}_crop.png"
        crop_img, _ = crop_board(image_path, weights=YOLO_WEIGHTS)
        cv2.imwrite(str(crop_path), crop_img)

        tmpl_path = Path(TEMPLATE_DIR) / f"{template_name}.png"
        draw_matches(tmpl_path, crop_path, pts_tmpl, pts_new, matches,
                     save_path=Path(RESULTS_DIR) / f"{name}_matches.png")
        draw_keypoints(crop_path, projected_kps,
                       save_path=Path(RESULTS_DIR) / f"{name}_pred.png")
