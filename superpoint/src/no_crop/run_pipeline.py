import os
import sys
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parents[2] / '.env')

sys.path.insert(0, str(Path(__file__).parent.parent / 'shared'))

from preprocessing import images_to_tensors
from superpoint import load_model, detect
from matcher import match, compute_homography
from predictor import project_keypoints
from visualization import draw_keypoints, draw_matches


# ── CONFIG ───────────────────────────────────────────────────────────────────

WEIGHTS            = os.getenv('SP_WEIGHTS')
TEMPLATE_DIR       = os.getenv('TEMPLATE_DIR')
TEST_IMAGE_DIR     = os.getenv('TEST_IMAGE_DIR')
RESULTS_DIR        = os.getenv('RESULTS_DIR')
CANONICAL_KPS_PATH = os.getenv('CANONICAL_KPS_PATH')


# ── HELPERS ──────────────────────────────────────────────────────────────────

def save_predicted_keypoints(name, projected_kps, results_dir):
    """Save predicted keypoints as .npy for later evaluation."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    save_path = results_dir / f"{name}_pred.npy"
    np.save(save_path, projected_kps)
    print(f"  Saved predicted kps -> {save_path}")


# ── RUN ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    # 1. Load model
    fe = load_model(WEIGHTS)

    # 2. Load canonical keypoints (clicked on template via keypoint_helper.py)
    canonical_kps = np.load(CANONICAL_KPS_PATH)
    print(f"Loaded {len(canonical_kps)} canonical keypoints from template")

    # 3. Load & detect on template (just the first one)
    template_tensors = images_to_tensors(TEMPLATE_DIR)
    template_name, template_tensor = next(iter(template_tensors.items()))
    pts_tmpl, desc_tmpl, _ = detect(fe, template_tensor)
    print(f"Template '{template_name}': {pts_tmpl.shape[1]} keypoints detected")

    # 4. Load new images and predict
    test_tensors = images_to_tensors(TEST_IMAGE_DIR)

    for name, tensor in test_tensors.items():
        print(f"\nProcessing: {name}")

        pts_new, desc_new, _ = detect(fe, tensor)

        matches = match(desc_tmpl, desc_new)
        H, mask = compute_homography(pts_tmpl, pts_new, matches)

        if H is None:
            print(f"  Skipping {name} — homography failed")
            continue

        print(f"  Matches: {len(matches)} | Inliers: {int(mask.sum())}")

        projected_kps = project_keypoints(canonical_kps, H)

        save_predicted_keypoints(name, projected_kps, RESULTS_DIR)

        img_path  = Path(TEST_IMAGE_DIR) / f"{name}.png"
        tmpl_path = Path(TEMPLATE_DIR) / f"{template_name}.png"

        draw_matches(tmpl_path, img_path, pts_tmpl, pts_new, matches,
                     save_path=Path(RESULTS_DIR) / f"{name}_matches.png")
        draw_keypoints(img_path, projected_kps,
                       save_path=Path(RESULTS_DIR) / f"{name}_pred.png")
