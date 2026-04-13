"""
Run this script once on your template image.
Left-click  -> add a keypoint
Right-click -> remove last keypoint
Press 's'   -> save keypoints to .npy file
Press 'q'   -> quit without saving
"""
import cv2
import numpy as np
import argparse
from pathlib import Path


keypoints = []


def mouse_callback(event, x, y, flags, param):
    img_display = param['img'].copy()

    if event == cv2.EVENT_LBUTTONDOWN:
        keypoints.append((x, y))
        print(f"Added keypoint {len(keypoints)}: ({x}, {y})")

    elif event == cv2.EVENT_RBUTTONDOWN:
        if keypoints:
            removed = keypoints.pop()
            print(f"Removed keypoint: {removed}")

    # Redraw all keypoints
    for i, (kx, ky) in enumerate(keypoints):
        cv2.circle(img_display, (kx, ky), 6, (0, 255, 0), -1)
        cv2.putText(img_display, str(i), (kx + 8, ky - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow("Click Keypoints", img_display)


def save_keypoints(image_path):
    if not keypoints:
        print("No keypoints to save.")
        return

    image_path = Path(image_path)
    save_path  = image_path.parent / f"{image_path.stem}_gt.npy"

    kps = np.array(keypoints, dtype=np.float32)
    np.save(save_path, kps)
    print(f"Saved {len(kps)} keypoints -> {save_path}")


def main(image_path):
    global keypoints
    keypoints = []          # reset for each image

    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Could not load image: {image_path}")
        return

    cv2.imshow("Click Keypoints", img)
    cv2.setMouseCallback("Click Keypoints", mouse_callback, param={'img': img})

    print(f"\nImage: {image_path}")
    print("Left-click to add | Right-click to remove last | 's' to save | 'q' to next/quit")

    while True:
        key = cv2.waitKey(20) & 0xFF

        if key == ord('s'):
            save_keypoints(image_path)
            break

        elif key == ord('q'):
            break

    cv2.destroyAllWindows()


def main_all(folder_path):
    folder_path = Path(folder_path)

    # supported_extensions = ('.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp', '.gif')
    supported_extensions = ('.png')

    # Skip files that already have ground truth saved
    images = sorted([
        p for p in folder_path.iterdir()
        if p.suffix.lower() in supported_extensions
        and not p.stem.endswith('_gt')
    ])

    if not images:
        print(f"No images found in: {folder_path}")
        return

    print(f"Found {len(images)} images in '{folder_path}'")

    for i, image_path in enumerate(images):
        gt_path = image_path.parent / f"{image_path.stem}_gt.npy"

        if gt_path.exists():
            print(f"[{i+1}/{len(images)}] Skipping '{image_path.name}' — gt already exists")
            continue

        print(f"\n[{i+1}/{len(images)}] Opening '{image_path.name}'")
        main(image_path)

    print("\nDone — all images processed.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--image',  type=str, help='Path to a single image file')
    group.add_argument('--all',    type=str, help='Path to folder — process all images in it')

    args = parser.parse_args()

    if args.image:
        main(args.image)
    elif args.all:
        main_all(args.all)