"""
Script: Project hex centers from ideal template → test image space
using homography computed from matched keypoints.

Outputs per test image:
  - <name>_hex.json        → hex_id: [x, y] in test image space
  - <name>_hex_viz.png     → hex centers drawn on test image

Usage:
    Single image:
        python project_hex.py --template_dir <path> --test_image <path> --output_dir <path>
    All images in directory:
        python project_hex.py --template_dir <path> --test_dir <path> --output_dir <path>
"""

import cv2
import numpy as np
import argparse
import json
from pathlib import Path


def load_template(template_dir):
    template_dir = Path(template_dir)

    kp_path  = template_dir / "cropped_ideal_image_gt.npy"
    hex_path = template_dir / "cropped_ideal_image_hex_centers.json"

    kps = np.load(str(kp_path))  # (N, 2)

    with open(hex_path, 'r') as f:
        hex_centers = json.load(f)  # { "hex_001": [x, y], ... }

    print(f"[Template] Keypoints: {kps.shape} | Hex centers: {len(hex_centers)}")
    return kps, hex_centers


def load_test(test_image_path):
    test_image_path = Path(test_image_path)
    pred_path = test_image_path.parent / (test_image_path.stem.replace("_crop", "_pred") + ".npy")

    img = cv2.imread(str(test_image_path))
    if img is None:
        raise FileNotFoundError(f"Could not load test image: {test_image_path}")

    kps = np.load(str(pred_path))  # (N, 2)
    print(f"[Test]     Loaded: {test_image_path.name} | Keypoints: {kps.shape}")
    return img, kps


def compute_homography(src_kps, dst_kps):
    H, mask = cv2.findHomography(src_kps, dst_kps, cv2.RANSAC, ransacReprojThreshold=5.0)
    if H is None:
        raise ValueError("Homography could not be computed.")
    inliers = int(mask.sum())
    print(f"[Homography] Inliers: {inliers} / {len(mask)}")
    return H, mask


def project_hex_centers(hex_centers, H):
    """
    Apply homography H to all hex center coordinates.
    Returns dict: hex_id → [x_proj, y_proj]
    """
    hex_ids = list(hex_centers.keys())
    pts = np.array([hex_centers[k] for k in hex_ids], dtype=np.float32)  # (M, 2)

    pts_h      = pts.reshape(-1, 1, 2)
    projected  = cv2.perspectiveTransform(pts_h, H).reshape(-1, 2)

    result = {
        hex_id: [round(float(x), 2), round(float(y), 2)]
        for hex_id, (x, y) in zip(hex_ids, projected)
    }
    return result


def save_json(projected_hex, save_path):
    with open(save_path, 'w') as f:
        json.dump(projected_hex, f, indent=2)
    print(f"[Saved] Hex JSON          → {save_path}")


def visualize_hex(test_img, projected_hex, template_kps, test_kps, mask, save_path):
    """
    Draw projected hex centers on the test image.
    Also draws the keypoints used for homography (inliers green, outliers red).
    """
    canvas = test_img.copy()

    for i, (px, py) in enumerate(test_kps):
        is_inlier = mask[i][0] == 1
        color = (0, 255, 0) if is_inlier else (0, 0, 255)
        cv2.circle(canvas, (int(px), int(py)), 6, color, -1)
        cv2.putText(canvas, f"kp{i}", (int(px) + 6, int(py) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

    for hex_id, (x, y) in projected_hex.items():
        x, y = int(x), int(y)
        cv2.circle(canvas, (x, y), 5, (0, 165, 255), -1)
        cv2.circle(canvas, (x, y), 5, (0, 0, 0), 1)
        cv2.putText(canvas, hex_id, (x + 6, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 165, 255), 1, cv2.LINE_AA)

    h = canvas.shape[0]
    cv2.putText(canvas, "Orange = Projected Hex Center", (10, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
    cv2.putText(canvas, "Green = Inlier KP | Red = Outlier KP", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imwrite(str(save_path), canvas)
    print(f"[Saved] Hex visualization → {save_path}")


def process_single(template_kps, hex_centers, test_image_path, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    test_img, test_kps = load_test(test_image_path)

    if len(template_kps) != len(test_kps):
        raise ValueError(
            f"Keypoint count mismatch: template={len(template_kps)}, test={len(test_kps)}"
        )

    H, mask = compute_homography(template_kps, test_kps)

    projected_hex = project_hex_centers(hex_centers, H)

    stem = Path(test_image_path).stem

    json_path = output_dir / f"{stem}_hex.json"
    viz_path  = output_dir / f"{stem}_hex_viz.png"

    save_json(projected_hex, json_path)
    visualize_hex(test_img, projected_hex, template_kps, test_kps, mask, viz_path)


def main():
    parser = argparse.ArgumentParser(description="Project hex centers from template → test image(s)")

    parser.add_argument('--template_dir', type=str, required=True,
                        help='Path to cropped_ideal_templates/ folder')
    parser.add_argument('--output_dir',   type=str, required=True,
                        help='Where to save JSON and visualization images')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--test_image', type=str,
                       help='Path to a single <name>_crop.png')
    group.add_argument('--test_dir',   type=str,
                       help='Path to folder containing *_crop.png files')

    args = parser.parse_args()

    template_kps, hex_centers = load_template(args.template_dir)

    if args.test_image:
        process_single(template_kps, hex_centers, args.test_image, args.output_dir)

    elif args.test_dir:
        test_dir    = Path(args.test_dir)
        crop_images = sorted(test_dir.glob("*_crop.png"))

        if not crop_images:
            print(f"No *_crop.png files found in: {test_dir}")
            return

        print(f"Found {len(crop_images)} test images in '{test_dir}'")

        for i, img_path in enumerate(crop_images):
            print(f"\n[{i+1}/{len(crop_images)}] Processing: {img_path.name}")
            try:
                process_single(template_kps, hex_centers, img_path, args.output_dir)
            except Exception as e:
                print(f"  ERROR: {e} — skipping.")

        print("\nDone.")


if __name__ == '__main__':
    main()
