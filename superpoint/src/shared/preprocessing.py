import os
from PIL import Image
import cv2
import numpy as np
import torch

Image.MAX_IMAGE_PIXELS = None

def convert_images_to_png(input_dir, output_dir=None):
    """Convert every image in input_dir to PNG format."""
    if output_dir is None:
        output_dir = input_dir

    os.makedirs(output_dir, exist_ok=True)

    supported_extensions = ('.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp', '.gif')

    for filename in os.listdir(input_dir):
        filepath = os.path.join(input_dir, filename)
        if not os.path.isfile(filepath):
            continue

        name, ext = os.path.splitext(filename)
        if ext.lower() in supported_extensions:
            img = Image.open(filepath)
            output_path = os.path.join(output_dir, name + '.png')
            img.save(output_path, 'PNG')
            img.close()
            os.remove(filepath)
            print(f"Converted: {filename} -> {name}.png")
        elif ext.lower() == '.png':
            print(f"Skipped (already PNG): {filename}")
        else:
            print(f"Skipped (unsupported): {filename}")

def convert_images_to_grayscale(input_dir, output_dir=None):
    """Convert every image in input_dir to grayscale."""
    if output_dir is None:
        output_dir = input_dir

    os.makedirs(output_dir, exist_ok=True)

    supported_extensions = ('.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp', '.gif', '.png')

    for filename in os.listdir(input_dir):
        filepath = os.path.join(input_dir, filename)
        if not os.path.isfile(filepath):
            continue

        name, ext = os.path.splitext(filename)
        if ext.lower() in supported_extensions:
            img = Image.open(filepath).convert('L')
            output_path = os.path.join(output_dir, filename)
            img.save(output_path)
            print(f"Converted to grayscale: {filename}")
        else:
            print(f"Skipped (unsupported): {filename}")

def print_unique_image_sizes(input_dir):
    """Print unique sizes (width x height) of all images in input_dir."""
    supported_extensions = ('.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp', '.gif', '.png')
    unique_sizes = set()

    for filename in os.listdir(input_dir):
        filepath = os.path.join(input_dir, filename)
        if not os.path.isfile(filepath):
            continue
        _, ext = os.path.splitext(filename)
        if ext.lower() in supported_extensions:
            img = Image.open(filepath)
            unique_sizes.add(img.size)

    print(f"Unique image sizes in '{input_dir}':")
    for size in unique_sizes:
        print(f"  {size[0]}x{size[1]}")

def resize_images(input_dir, output_dir=None, max_long_side=960):
    """Resize every image in input_dir to superpoint criteria.
    Preserves orientation: landscape stays landscape, portrait stays portrait."""
    if output_dir is None:
        output_dir = input_dir

    os.makedirs(output_dir, exist_ok=True)

    supported_extensions = ('.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp', '.gif', '.png')

    for filename in os.listdir(input_dir):
        filepath = os.path.join(input_dir, filename)
        if not os.path.isfile(filepath):
            continue

        name, ext = os.path.splitext(filename)
        if ext.lower() in supported_extensions:
            img = cv2.imread(filepath)
            if img is None:
                continue
            h, w = img.shape[:2]

            is_landscape = w >= h  # True = landscape, False = portrait

            if max(h, w) <= max_long_side:
                print(f"Skipped (already small enough): {filename}  {w}x{h}")
                continue

            scale = max_long_side / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)

            # Ensure orientation is preserved after int rounding
            if is_landscape and new_h > new_w:
                new_w, new_h = new_h, new_w
            elif not is_landscape and new_w > new_h:
                new_w, new_h = new_h, new_w

            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, resized)
            orientation = "landscape" if is_landscape else "portrait"
            print(f"Resized ({orientation}): {filename}  {w}x{h}  →  {new_w}x{new_h}")
        else:
            print(f"Skipped (not resized): {filename}")

def normalize_images(input_dir, output_dir=None):
    """Normalize pixel values to [0, 1] range and save as .npy files.
    SuperPoint expects float tensors in this range."""
    if output_dir is None:
        output_dir = input_dir

    os.makedirs(output_dir, exist_ok=True)

    supported_extensions = ('.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp', '.gif', '.png')

    for filename in os.listdir(input_dir):
        filepath = os.path.join(input_dir, filename)
        if not os.path.isfile(filepath):
            continue

        name, ext = os.path.splitext(filename)
        if ext.lower() in supported_extensions:
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            normalized = img.astype(np.float32) / 255.0  # [0, 1] range
            output_path = os.path.join(output_dir, name + '.npy')
            np.save(output_path, normalized)
            print(f"Normalized: {filename} -> {name}.npy  (shape: {normalized.shape})")
        else:
            print(f"Skipped (unsupported): {filename}")

def images_to_tensors(input_dir):
    """Load precomputed .npy files from input_dir and convert each to a PyTorch tensor 
    with shape (1, 1, H, W). Falls back to normalizing from image if .npy not found."""
    supported_extensions = ('.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp', '.gif', '.png')
    tensors = {}

    for filename in os.listdir(input_dir):
        filepath = os.path.join(input_dir, filename)
        if not os.path.isfile(filepath):
            continue

        name, ext = os.path.splitext(filename)
        if ext.lower() not in supported_extensions:
            continue

        npy_path = os.path.join(input_dir, name + '.npy')

        if os.path.exists(npy_path):
            normalized = np.load(npy_path)  # already float32 in [0, 1]
            print(f"Loaded precomputed: {name}.npy")
        else:
            print(f"No .npy found, normalizing from image: {filename}")
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Could not load image: {filename}, skipping.")
                continue
            normalized = img.astype(np.float32) / 255.0

        tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        tensors[name] = tensor
        print(f"  shape: {tensor.shape}  dtype: {tensor.dtype}  range: [{tensor.min():.2f}, {tensor.max():.2f}]")

    return tensors

if __name__ == '__main__':
    # input_dir = "../raw_dataset/preprocessed/ideal_templates"
    # convert_images_to_png(input_dir=input_dir)
    # convert_images_to_grayscale(input_dir=input_dir)
    # print_unique_image_sizes(input_dir=input_dir)
    # resize_images(input_dir=input_dir)
    # normalize_images(input_dir=input_dir)
    # images_to_tensors(input_dir=input_dir)
    # input_dir = "../raw_dataset/preprocessed/test_images"
    # convert_images_to_png(input_dir=input_dir)
    # convert_images_to_grayscale(input_dir=input_dir)
    # print_unique_image_sizes(input_dir=input_dir)
    # resize_images(input_dir=input_dir)
    # normalize_images(input_dir=input_dir)
    # images_to_tensors(input_dir=input_dir)

    input_dir = "../raw_dataset/preprocessed/cropped_ideal_templates"
    normalize_images(input_dir=input_dir)

