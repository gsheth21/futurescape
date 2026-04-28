import argparse
import os
import cv2
import numpy as np
from pathlib import Path

COLORS = [
    (0, 0, 255),
    (0, 255, 0),
    (255, 0, 0),
    (0, 255, 255),
    (255, 0, 255),
    (255, 165, 0),
    (128, 0, 128),
    (0, 128, 128),
]


def load_classes(classes_file):
    p = Path(classes_file)
    if p.exists():
        return [line.strip() for line in p.read_text().splitlines() if line.strip()]
    return []


def yolo_to_pixel(cx, cy, w, h, img_w, img_h):
    x1 = int((cx - w / 2) * img_w)
    y1 = int((cy - h / 2) * img_h)
    x2 = int((cx + w / 2) * img_w)
    y2 = int((cy + h / 2) * img_h)
    return x1, y1, x2, y2


def draw_labels(image, label_path, classes):
    img_h, img_w = image.shape[:2]
    if not Path(label_path).exists():
        print(f"Label file not found: {label_path}")
        return image

    for line in Path(label_path).read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_id = int(parts[0])
        cx, cy, w, h = map(float, parts[1:5])
        x1, y1, x2, y2 = yolo_to_pixel(cx, cy, w, h, img_w, img_h)

        color = COLORS[cls_id % len(COLORS)]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        label = classes[cls_id] if cls_id < len(classes) else str(cls_id)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(image, label, (x1, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return image


def show_single(image_path, label_path, classes_file=None):
    image_path = Path(image_path)
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Could not read image: {image_path}")
        return

    # Auto-detect label if not provided
    if label_path is None:
        label_path = image_path.parent.parent / "labels" / (image_path.stem + ".txt")
        print(f"Auto-detected label: {label_path}")

    # Auto-detect classes.txt if not provided
    classes = []
    if classes_file:
        classes = load_classes(classes_file)
    else:
        auto_classes = image_path.parent.parent / "classes.txt"
        if auto_classes.exists():
            classes = load_classes(auto_classes)

    image = draw_labels(image, label_path, classes)

    # Resize to fit screen
    # max_w, max_h = 1280, 720
    # h, w = image.shape[:2]
    # scale = min(max_w / w, max_h / h, 1.0)
    # if scale < 1.0:
    #     image = cv2.resize(image, (int(w * scale), int(h * scale)))

    cv2.namedWindow(image_path.name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(image_path.name, 1280, 720)
    cv2.imshow(image_path.name, image)
    print("Press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def visualize_dataset(dataset_dir, output_dir):
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    classes = load_classes(dataset_dir / "classes.txt")
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"

    image_files = sorted(
        p for p in images_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    if not image_files:
        print(f"No images found in {images_dir}")
        return

    print(f"Found {len(image_files)} images. Classes: {classes}")

    for img_path in image_files:
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  Could not read {img_path.name}, skipping.")
            continue

        label_path = labels_dir / (img_path.stem + ".txt")
        image = draw_labels(image, label_path, classes)

        out_path = output_dir / img_path.name
        cv2.imwrite(str(out_path), image)
        print(f"  Saved: {out_path}")

    print(f"\nDone. {len(image_files)} images saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize YOLO labels on images")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Single image mode
    show_parser = subparsers.add_parser("show", help="Display a single image with labels")
    show_parser.add_argument("image", help="Path to image file")
    show_parser.add_argument("--label", default=None, help="Path to label .txt file (auto-detected if omitted)")
    show_parser.add_argument("--classes", default=None, help="Path to classes.txt (auto-detected if omitted)")

    # Batch mode
    batch_parser = subparsers.add_parser("batch", help="Annotate all images in a dataset dir and save")
    batch_parser.add_argument("dataset_dir", help="Path to dataset dir (contains images/, labels/, classes.txt)")
    batch_parser.add_argument("output_dir", help="Path to save annotated images")

    args = parser.parse_args()

    if args.mode == "show":
        show_single(args.image, args.label, args.classes)
    elif args.mode == "batch":
        visualize_dataset(args.dataset_dir, args.output_dir)