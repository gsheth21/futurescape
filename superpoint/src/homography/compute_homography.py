"""
Script 1: Compute homography between ideal template and test image(s),
and visualize:
  1. Keypoint matches (inliers green, outliers red)
  2. Warped template overlaid on test image

Usage:
    Single image:
        python homography_viz.py --template_dir <path> --test_image <path_to_crop.png> --output_dir <path>
    All images in directory:
        python homography_viz.py --template_dir <path> --test_dir <path> --output_dir <path>
"""

import cv2
import numpy as np
import argparse
from pathlib import Path


def load_template(template_dir):
    template_dir = Path(template_dir)

    img_path = template_dir / "cropped_ideal_image.png"
    kp_path  = template_dir / "cropped_ideal_image_gt.npy"

    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Could not load template image: {img_path}")

    kps = np.load(str(kp_path))  # (N, 2)
    print(f"[Template] Loaded image: {img_path.name} | Keypoints: {kps.shape}")
    return img, kps


def load_test(test_image_path):
    test_image_path = Path(test_image_path)

    # Expect <name>_crop.png and <name>_pred.npy
    pred_path = test_image_path.parent / (test_image_path.stem.replace("_crop", "_pred") + ".npy")

    img = cv2.imread(str(test_image_path))
    if img is None:
        raise FileNotFoundError(f"Could not load test image: {test_image_path}")

    kps = np.load(str(pred_path))  # (N, 2)
    print(f"[Test]     Loaded image: {test_image_path.name} | Keypoints: {kps.shape}")
    return img, kps


def compute_homography(src_kps, dst_kps):
    """
    Compute homography from src (template) to dst (test image) using RANSAC.
    Returns H and mask (inlier/outlier per point).
    """
    H, mask = cv2.findHomography(src_kps, dst_kps, cv2.RANSAC, ransacReprojThreshold=5.0)
    if H is None:
        raise ValueError("Homography could not be computed.")
    inliers  = int(mask.sum())
    outliers = len(mask) - inliers
    print(f"[Homography] Inliers: {inliers} | Outliers: {outliers}")
    return H, mask


def visualize_matches(template_img, template_kps, test_img, test_kps, mask, save_path):
    """
    Draw side-by-side match visualization.
    Inliers  → green lines
    Outliers → red lines
    """
    h1, w1 = template_img.shape[:2]
    h2, w2 = test_img.shape[:2]

    canvas_h = max(h1, h2)
    canvas_w = w1 + w2
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:h1, :w1]       = template_img
    canvas[:h2, w1:w1+w2]  = test_img

    for i, ((tx, ty), (px, py)) in enumerate(zip(template_kps, test_kps)):
        tx, ty = int(tx), int(ty)
        px, py = int(px), int(py)
        is_inlier = mask[i][0] == 1

        color = (0, 255, 0) if is_inlier else (0, 0, 255)

        pt1 = (tx, ty)
        pt2 = (w1 + px, py)

        cv2.line(canvas, pt1, pt2, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, pt1, 5, color, -1)
        cv2.circle(canvas, pt2, 5, color, -1)

        # Label index
        cv2.putText(canvas, str(i), (tx - 10, ty - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
        cv2.putText(canvas, str(i), (w1 + px + 5, py - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    # Legend
    cv2.putText(canvas, "Template", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(canvas, "Test Image", (w1 + 10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(canvas, "Green=Inlier  Red=Outlier", (10, canvas_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imwrite(str(save_path), canvas)
    print(f"[Saved] Match visualization → {save_path}")


def visualize_warp(template_img, test_img, H, save_path):
    """
    Warp template onto test image and blend them for overlay visualization.
    """
    h, w = test_img.shape[:2]

    warped = cv2.warpPerspective(template_img, H, (w, h))

    # Create mask of warped region (non-black pixels)
    gray_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    _, warp_mask = cv2.threshold(gray_warped, 1, 255, cv2.THRESH_BINARY)
    warp_mask_3ch = cv2.merge([warp_mask, warp_mask, warp_mask])

    # Blend: 50% test image + 50% warped template
    blended = cv2.addWeighted(test_img, 0.5, warped, 0.5, 0)

    # Where warped has no content, show original test image
    result = np.where(warp_mask_3ch > 0, blended, test_img)

    # Draw border of warped region
    contours, _ = cv2.findContours(warp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, (0, 255, 255), 2)

    cv2.imwrite(str(save_path), result)
    print(f"[Saved] Warp overlay      → {save_path}")


def process_single(template_img, template_kps, test_image_path, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    test_img, test_kps = load_test(test_image_path)

    if len(template_kps) != len(test_kps):
        raise ValueError(
            f"Keypoint count mismatch: template={len(template_kps)}, test={len(test_kps)}"
        )

    H, mask = compute_homography(template_kps, test_kps)

    stem = Path(test_image_path).stem  # e.g. "1-2030_crop"

    match_path = output_dir / f"{stem}_matches.png"
    warp_path  = output_dir / f"{stem}_warp.png"

    visualize_matches(template_img, template_kps, test_img, test_kps, mask, match_path)
    visualize_warp(template_img, test_img, H, warp_path)

    return H


def main():
    parser = argparse.ArgumentParser(description="Homography visualization: template → test image(s)")

    parser.add_argument('--template_dir', type=str, required=True,
                        help='Path to cropped_ideal_templates/ folder')
    parser.add_argument('--output_dir',   type=str, required=True,
                        help='Where to save visualization images')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--test_image', type=str,
                       help='Path to a single <name>_crop.png')
    group.add_argument('--test_dir',   type=str,
                       help='Path to folder containing *_crop.png files')

    args = parser.parse_args()

    template_img, template_kps = load_template(args.template_dir)

    if args.test_image:
        process_single(template_img, template_kps, args.test_image, args.output_dir)

    elif args.test_dir:
        test_dir = Path(args.test_dir)
        crop_images = sorted(test_dir.glob("*_crop.png"))

        if not crop_images:
            print(f"No *_crop.png files found in: {test_dir}")
            return

        print(f"Found {len(crop_images)} test images in '{test_dir}'")

        for i, img_path in enumerate(crop_images):
            print(f"\n[{i+1}/{len(crop_images)}] Processing: {img_path.name}")
            try:
                process_single(template_img, template_kps, img_path, args.output_dir)
            except Exception as e:
                print(f"  ERROR: {e} — skipping.")

        print("\nDone.")


if __name__ == '__main__':
    main()