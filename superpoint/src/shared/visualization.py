import cv2
import numpy as np


def draw_keypoints(image_path, keypoints, color=(0, 255, 0), radius=6, save_path=None):
    """
    Draw keypoints on an image.

    Args:
        image_path : path to the image file
        keypoints  : (N, 2) array of x,y coordinates
        color      : BGR color tuple
        radius     : circle radius
        save_path  : if given, saves the result image there
    """
    img = cv2.imread(str(image_path))

    for (x, y) in keypoints:
        cv2.circle(img, (int(x), int(y)), radius, color, -1)

    if save_path:
        cv2.imwrite(str(save_path), img)
        print(f"Saved: {save_path}")

    # cv2.imshow("Keypoints", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def draw_matches(image_path_1, image_path_2, pts1, pts2, matches, save_path=None):
    """
    Draw match lines between two images side by side.

    Args:
        image_path_1 : path to first image  (template)
        image_path_2 : path to second image (test/new)
        pts1         : (3, N1) keypoints from image 1
        pts2         : (3, N2) keypoints from image 2
        matches      : list of (idx1, idx2) pairs
        save_path    : optional save path
    """
    img1 = cv2.imread(str(image_path_1))
    img2 = cv2.imread(str(image_path_2))

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1]       = img1
    canvas[:h2, w1:w1+w2]  = img2

    for idx1, idx2 in matches:
        x1, y1 = int(pts1[0, idx1]), int(pts1[1, idx1])
        x2, y2 = int(pts2[0, idx2]) + w1, int(pts2[1, idx2])
        cv2.line(canvas, (x1, y1), (x2, y2), (0, 255, 255), 1)
        cv2.circle(canvas, (x1, y1), 3, (0, 255, 0), -1)
        cv2.circle(canvas, (x2, y2), 3, (0, 0, 255), -1)

    if save_path:
        cv2.imwrite(str(save_path), canvas)
        print(f"Saved: {save_path}")

    # cv2.imshow("Matches", canvas)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def draw_pck_thresholds(image_path, gt_kps, pred_kps, thresholds=[5, 10, 20, 50], save_path=None):
    """
    For each keypoint, draw:
      - threshold circles (rings) around ground truth
      - ground truth point  (green)
      - predicted point     (red)
      - line connecting them

    Args:
        image_path  : path to the image
        gt_kps      : (N, 2) ground truth keypoints
        pred_kps    : (N, 2) predicted keypoints
        thresholds  : list of PCK threshold radii to draw
        save_path   : optional save path
    """
    img = cv2.imread(str(image_path))

    # One distinct color per threshold ring (outermost to innermost)
    threshold_colors = {
        50 : (255, 200,   0),   # blue
        20 : (255, 100, 100),   # purple
        10 : (  0, 165, 255),   # orange
        5  : (  0,   0, 255),   # red
    }

    for i, (gt, pred) in enumerate(zip(gt_kps, pred_kps)):
        gt_x,   gt_y   = int(gt[0]),   int(gt[1])
        pred_x, pred_y = int(pred[0]), int(pred[1])

        # Draw threshold rings around ground truth (largest first so smaller ones stay visible)
        for tau in sorted(thresholds, reverse=True):
            color = threshold_colors.get(tau, (200, 200, 200))
            cv2.circle(img, (gt_x, gt_y), tau, color, 1)

        # Line from predicted -> ground truth
        cv2.line(img, (pred_x, pred_y), (gt_x, gt_y), (200, 200, 200), 1)

        # Ground truth point  (green filled)
        cv2.circle(img, (gt_x, gt_y), 2, (0, 255, 0), -1)

        # Predicted point  (red filled)
        cv2.circle(img, (pred_x, pred_y), 2, (0, 0, 255), -1)

        # Keypoint index label
        cv2.putText(img, str(i), (gt_x + 6, gt_y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # Draw legend in top-left corner
    legend_items = [
        ("GT keypoint",       (0, 255, 0)),
        ("Pred keypoint",     (0, 0, 255)),
    ] + [(f"PCK@{tau} radius", threshold_colors.get(tau, (200,200,200)))
         for tau in sorted(thresholds)]

    for idx, (label, color) in enumerate(legend_items):
        y = 20 + idx * 22
        cv2.circle(img, (15, y), 6, color, -1 if idx < 2 else 1)
        cv2.putText(img, label, (28, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    if save_path:
        cv2.imwrite(str(save_path), img)
        print(f"Saved: {save_path}")

    cv2.imshow("PCK Thresholds", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

"""
Visualize PCK thresholds for a single image.

Usage:
    python visualize_pck.py --image <image_path> --gt <gt_npy> --pred <pred_npy>

Example:
    python visualize_pck.py \
        --image ../raw_dataset/preprocessed/test_images/img1.png \
        --gt    ../raw_dataset/preprocessed/test_images/img1_gt.npy \
        --pred  ../raw_dataset/preprocessed/test_images_results/img1_pred.npy
"""

import numpy as np
import argparse
from pathlib import Path
from visualization import draw_pck_thresholds


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--gt',    type=str, required=True)
    parser.add_argument('--pred',  type=str, required=True)
    args = parser.parse_args()

    gt_kps   = np.load(args.gt)
    pred_kps = np.load(args.pred)

    save_path = Path(args.image).parent / f"{Path(args.image).stem}_pck_viz.png"

    draw_pck_thresholds(
        image_path = args.image,
        gt_kps     = gt_kps,
        pred_kps   = pred_kps,
        thresholds = [5, 10, 20, 50],
        save_path  = save_path
    )