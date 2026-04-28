"""
Loads ground truth and predicted keypoints and computes evaluation metrics.

Expected file structure:
    CROPPED_GT_DIR/
        img1_gt.npy             <- ground truth clicked on FULL images via keypoint_helper.py
    CROPPED_PRED_DIR/
        img1_pred.npy           <- predicted by run_pipeline_cropped.py (in crop space)
        img1_bbox.npy           <- bbox saved by run_pipeline_cropped.py (x1, y1, x2, y2)

Usage:
    python cropped_eval.py
"""

import os
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parents[2] / '.env')


# ── CONFIG ───────────────────────────────────────────────────────────────────

GT_DIR   = os.getenv('CROPPED_GT_DIR')
PRED_DIR = os.getenv('CROPPED_PRED_DIR')

PCK_THRESHOLDS = [5, 10, 20, 50]


# ── METRICS ──────────────────────────────────────────────────────────────────

def euclidean_distances(gt_kps, pred_kps):
    return np.linalg.norm(gt_kps - pred_kps, axis=1)


def mean_euclidean_distance(distances):
    return float(np.mean(distances))


def median_euclidean_distance(distances):
    return float(np.median(distances))


def pck(distances, threshold):
    return float(np.mean(distances < threshold) * 100)


def per_keypoint_mean_error(all_distances):
    stacked = np.stack(all_distances, axis=0)
    return np.mean(stacked, axis=0)


# ── LOAD PAIRS ───────────────────────────────────────────────────────────────

def load_gt_pred_pairs(gt_dir, pred_dir):
    """
    Find matching gt/pred .npy pairs and load them.
    GT is automatically adjusted from full image space to crop space using saved bbox.

    Returns:
        pairs         : list of (name, gt_kps, pred_kps)
        total_gt      : total number of GT files found
        yolo_detected : number of images for which YOLO bbox was found
    """
    gt_dir   = Path(gt_dir)
    pred_dir = Path(pred_dir)

    total_gt      = 0
    yolo_detected = 0

    pairs = []
    for gt_path in sorted(gt_dir.glob('*_gt.npy')):
        total_gt += 1
        name      = gt_path.stem.replace('_gt', '')
        pred_path = pred_dir / f"{name}_pred.npy"
        bbox_path = pred_dir / f"{name}_bbox.npy"

        if not bbox_path.exists():
            print(f"  Warning: no bbox found for '{name}' — YOLO likely failed, skipping")
            continue

        yolo_detected += 1

        if not pred_path.exists():
            print(f"  Warning: no prediction found for '{name}', skipping")
            continue

        gt_kps   = np.load(gt_path)
        pred_kps = np.load(pred_path)

        # Auto-adjust GT from full image space -> crop space
        x1, y1, _, _ = np.load(bbox_path)
        gt_kps = gt_kps - np.array([x1, y1], dtype=np.float32)

        if gt_kps.shape != pred_kps.shape:
            print(f"  Warning: shape mismatch for '{name}' "
                  f"gt={gt_kps.shape} pred={pred_kps.shape}, skipping")
            continue

        pairs.append((name, gt_kps, pred_kps))
        print(f"  Loaded pair: '{name}' | {len(gt_kps)} keypoints")

    return pairs, total_gt, yolo_detected


# ── REPORT ───────────────────────────────────────────────────────────────────

def print_report(pairs, total_gt, yolo_detected):
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
    num_evaluated = len(pairs)
    num_kps       = len(all_distances[0]) if all_distances else 0

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(f"\n  Total GT images          : {total_gt}")
    print(f"  YOLO board detected      : {yolo_detected} / {total_gt}")
    print(f"  Successfully evaluated   : {num_evaluated} / {total_gt}")
    print(f"  Keypoints per image      : {num_kps}")

    print(f"\n  Mean Distance            : {mean_euclidean_distance(all_dist_flat):.2f} px")
    print(f"  Median Distance          : {median_euclidean_distance(all_dist_flat):.2f} px")

    print(f"\n  {'Threshold':<12} {'% Correct':>10}   {'Images all-KP correct':>22}")
    print(f"  {'-'*48}")
    for tau in PCK_THRESHOLDS:
        pck_pct = pck(all_dist_flat, tau)
        imgs_all_correct = sum(1 for d in all_distances if np.all(d < tau))
        print(f"  PCK@{tau:<8} {pck_pct:>9.1f}%   {imgs_all_correct:>12} / {num_evaluated}")

    per_kp = per_keypoint_mean_error(all_distances)
    print(f"\n  {'Keypoint':<12} {'Mean Error (px)':>16}")
    print(f"  {'-'*30}")
    for i, err in enumerate(per_kp):
        print(f"  KP {i:>2}        {err:>14.2f} px")

    print("=" * 60)


# ── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("Loading gt/pred pairs...")
    pairs, total_gt, yolo_detected = load_gt_pred_pairs(GT_DIR, PRED_DIR)

    if not pairs:
        print("No valid pairs found. Run keypoint_helper.py and run_pipeline_cropped.py first.")
    else:
        print_report(pairs, total_gt, yolo_detected)
