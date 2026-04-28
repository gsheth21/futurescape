"""
Loads ground truth and predicted keypoints and computes evaluation metrics.

Expected file structure:
    GT_DIR/
        img1_gt.npy       <- ground truth clicked via keypoint_helper.py
        img2_gt.npy
    PRED_DIR/
        img1_pred.npy     <- predicted by run_pipeline.py
        img2_pred.npy

Usage:
    python evaluator.py
"""

import os
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parents[2] / '.env')


# ── CONFIG ───────────────────────────────────────────────────────────────────

GT_DIR   = os.getenv('GT_DIR')
PRED_DIR = os.getenv('PRED_DIR')

# PCK thresholds in pixels
PCK_THRESHOLDS = [5, 10, 20, 50]


# ── METRICS ──────────────────────────────────────────────────────────────────

def euclidean_distances(gt_kps, pred_kps):
    """
    Per-keypoint Euclidean distance between ground truth and predicted.

    Args:
        gt_kps   : (N, 2)
        pred_kps : (N, 2)

    Returns:
        distances : (N,) array of pixel errors
    """
    return np.linalg.norm(gt_kps - pred_kps, axis=1)


def mean_euclidean_distance(distances):
    return float(np.mean(distances))


def median_euclidean_distance(distances):
    return float(np.median(distances))


def pck(distances, threshold):
    """Percentage of keypoints with error < threshold pixels."""
    return float(np.mean(distances < threshold) * 100)


def per_keypoint_mean_error(all_distances):
    """
    Mean error per keypoint index across all images.

    Args:
        all_distances : list of (N,) arrays, one per image

    Returns:
        per_kp_error : (N,) mean error for each keypoint index
    """
    stacked = np.stack(all_distances, axis=0)  # (num_images, N)
    return np.mean(stacked, axis=0)            # (N,)


# ── LOAD PAIRS ───────────────────────────────────────────────────────────────

def load_gt_pred_pairs(gt_dir, pred_dir):
    """
    Find matching gt/pred .npy pairs and load them.

    Returns:
        pairs : list of (name, gt_kps, pred_kps)
    """
    gt_dir   = Path(gt_dir)
    pred_dir = Path(pred_dir)

    pairs = []
    for gt_path in sorted(gt_dir.glob('*_gt.npy')):
        name      = gt_path.stem.replace('_gt', '')
        pred_path = pred_dir / f"{name}_pred.npy"

        if not pred_path.exists():
            print(f"  Warning: no prediction found for '{name}', skipping")
            continue

        gt_kps   = np.load(gt_path)
        pred_kps = np.load(pred_path)

        if gt_kps.shape != pred_kps.shape:
            print(f"  Warning: shape mismatch for '{name}' "
                  f"gt={gt_kps.shape} pred={pred_kps.shape}, skipping")
            continue

        pairs.append((name, gt_kps, pred_kps))
        print(f"  Loaded pair: '{name}' | {len(gt_kps)} keypoints")

    return pairs


# ── REPORT ───────────────────────────────────────────────────────────────────

def print_report(pairs):
    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)

    all_distances = []

    for name, gt_kps, pred_kps in pairs:
        distances = euclidean_distances(gt_kps, pred_kps)
        all_distances.append(distances)

        print(f"\nImage: {name}")
        print(f"  Mean Distance   : {mean_euclidean_distance(distances):.2f} px")
        print(f"  Median Distance : {median_euclidean_distance(distances):.2f} px")
        for tau in PCK_THRESHOLDS:
            print(f"  PCK@{tau:<4}        : {pck(distances, tau):.1f}%")

    all_dist_flat = np.concatenate(all_distances)

    print("\n" + "-" * 60)
    print("OVERALL (all images combined)")
    print(f"  Mean Distance   : {mean_euclidean_distance(all_dist_flat):.2f} px")
    print(f"  Median Distance : {median_euclidean_distance(all_dist_flat):.2f} px")
    for tau in PCK_THRESHOLDS:
        print(f"  PCK@{tau:<4}        : {pck(all_dist_flat, tau):.1f}%")

    per_kp = per_keypoint_mean_error(all_distances)
    print("\n" + "-" * 60)
    print("PER KEYPOINT MEAN ERROR")
    for i, err in enumerate(per_kp):
        print(f"  Keypoint {i:>2}     : {err:.2f} px")

    print("=" * 60)


# ── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("Loading gt/pred pairs...")
    pairs = load_gt_pred_pairs(GT_DIR, PRED_DIR)

    if not pairs:
        print("No valid pairs found. Run keypoint_helper.py and run_pipeline.py first.")
    else:
        print_report(pairs)
