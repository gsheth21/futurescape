"""
Mark hex centers on a board image and export to JSON.
Left-click  -> add a hex center
Right-click -> remove last hex center
Press 's'   -> save to JSON file
Press 'z'   -> undo last point
Press 'q'   -> quit without saving
"""
import cv2
import numpy as np
import argparse
import json
from pathlib import Path


hex_centers = []


def mouse_callback(event, x, y, flags, param):
    img_display = param['img'].copy()

    if event == cv2.EVENT_LBUTTONDOWN:
        hex_centers.append((x, y))
        print(f"Added hex_{len(hex_centers):03d}: ({x}, {y})")

    elif event == cv2.EVENT_RBUTTONDOWN:
        if hex_centers:
            removed = hex_centers.pop()
            print(f"Removed last hex center: {removed}")

    redraw(img_display)


def redraw(img_display):
    for i, (hx, hy) in enumerate(hex_centers):
        label = f"hex_{i+1:03d}"
        cv2.circle(img_display, (hx, hy), 8, (0, 255, 0), -1)
        cv2.circle(img_display, (hx, hy), 8, (0, 0, 0), 1)
        cv2.putText(img_display, label, (hx + 10, hy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.imshow("Mark Hex Centers", img_display)


def save_json(image_path):
    if not hex_centers:
        print("No hex centers to save.")
        return

    image_path = Path(image_path)
    save_path = image_path.parent / f"{image_path.stem}_hex_centers.json"

    data = {
        f"hex_{i+1:03d}": list(pt)
        for i, pt in enumerate(hex_centers)
    }

    with open(save_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Saved {len(data)} hex centers -> {save_path}")


def main(image_path):
    global hex_centers
    hex_centers = []

    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Could not load image: {image_path}")
        return

    cv2.imshow("Mark Hex Centers", img)
    cv2.setMouseCallback("Mark Hex Centers", mouse_callback, param={'img': img})

    print(f"\nImage: {image_path}")
    print("Left-click to add | Right-click to remove last | 'z' to undo | 's' to save | 'q' to quit")

    while True:
        key = cv2.waitKey(20) & 0xFF

        if key == ord('s'):
            save_json(image_path)
            break

        elif key == ord('z'):
            if hex_centers:
                removed = hex_centers.pop()
                print(f"Undo: removed {removed}")
                img_copy = img.copy()
                redraw(img_copy)

        elif key == ord('q'):
            print("Quit without saving.")
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Manually mark hex centers on a board image.")
    parser.add_argument('--image', type=str, required=True, help='Path to the board image')
    args = parser.parse_args()

    main(args.image)