"""
Combines multiple piece-class subdirectories from the dataset folder into a
single merged YOLO-format dataset.

Each subdirectory must contain a folder matching "futures workshop 2024"
(handles the "Trasforming" / "Transforming" typo) with images/, labels/,
and notes.json inside it.

Class IDs are read from notes.json (supports multiple categories per source).
Global IDs are assigned serially in the order classes are first encountered
across sources (sorted by subdir name for determinism).

Output structure:
    <OUTPUT_DIR>/
        images/     ── one image per unique original image name
        labels/     ── merged YOLO annotations (all classes) per image
        classes.txt ── global class list, one name per line
        notes.json  ── global category list in Label Studio format
"""

import json
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path
from dotenv import load_dotenv

# Load machine-specific paths from piece_detection/.env
load_dotenv(Path(__file__).parents[2] / ".env")

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
DATASET_DIR = Path(os.environ["PIECES_DATASET_DIR"])
OUTPUT_DIR  = Path(os.getenv("ALL_PIECES_DATASET_DIR", str(DATASET_DIR / "all_pieces")))

# Class aliases: any name in the key will be treated as the value
CLASS_ALIASES: dict[str, str] = {
    "blue_fish": "aquatic_life",
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def load_local_classes(notes_path: Path) -> dict[int, str]:
    """
    Parse notes.json and return {local_id: class_name} for all categories.
    """
    data = json.loads(notes_path.read_text())
    return {cat["id"]: cat["name"] for cat in data.get("categories", [])}


def original_stem(filename: str) -> str:
    """
    Strip the leading random code from a filename stem.

    '02f1a648-9-2060.jpg' → '9-2060'
    'cd23d1bc-Grad-2050.jpg' → 'Grad-2050'
    """
    stem = Path(filename).stem            # drop extension
    return stem[stem.index("-") + 1:]     # drop everything up to and including first '-'


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    # ── 1. Discover all sources ──────────────────────────────────────────── #
    # sources: list of (workshop_dir, local_id_to_name)
    sources: list[tuple[Path, dict[int, str]]] = []

    print("Scanning sources...")
    for subdir in sorted(DATASET_DIR.iterdir()):
        if not subdir.is_dir() or subdir.name == OUTPUT_DIR.name:
            continue
        notes_file = subdir / "notes.json"
        if not notes_file.exists():
            print(f"  WARNING: no notes.json in '{subdir}' — skipping")
            continue
        local_classes = load_local_classes(notes_file)
        local_classes = {lid: CLASS_ALIASES.get(name, name) for lid, name in local_classes.items()}
        sources.append((subdir, local_classes))
        print(f"  {subdir.name!r}  →  {local_classes}")

    if not sources:
        print("No valid sources found. Check DATASET_DIR.")
        return

    # ── 2. Build global class map (serial, first-encountered order) ──────── #
    # Walk sources in the same sorted order to keep IDs deterministic.
    global_class_id: dict[str, int] = {}
    for _, local_classes in sources:
        for local_id in sorted(local_classes):          # stable within source
            name = local_classes[local_id]
            if name not in global_class_id:
                global_class_id[name] = len(global_class_id)

    print("\nGlobal class map:")
    for cls_name, cls_idx in global_class_id.items():
        print(f"  {cls_idx}: {cls_name}")

    # ── 3. Index every labeled image by its original stem ────────────────── #
    # orig_stem → list of (image_path, label_path, local_to_global: dict[int,int])
    index: dict[str, list[tuple[Path, Path, dict[int, int]]]] = defaultdict(list)

    collision_count = 0
    for workshop_dir, local_classes in sources:
        # per-source remapping table: local class id → global class id
        local_to_global: dict[int, int] = {
            lid: global_class_id[name] for lid, name in local_classes.items()
        }

        img_dir = workshop_dir / "images"
        lbl_dir = workshop_dir / "labels"
        seen: dict[str, int] = defaultdict(int)

        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if not lbl_path.exists():
                print(f"  WARNING: no label for '{img_path.name}' — skipping")
                continue

            orig = original_stem(img_path.name)
            seen[orig] += 1
            if seen[orig] > 1:
                collision_count += 1
                print(
                    f"  COLLISION: '{orig}' appears {seen[orig]}x "
                    f"in '{workshop_dir.name}' — all labels collected"
                )
            index[orig].append((img_path, lbl_path, local_to_global))

    print(
        f"\n{len(index)} unique original images"
        + (f"  ({collision_count} intra-source name collision(s))" if collision_count else "")
    )

    # ── 4. Write combined dataset ─────────────────────────────────────────── #
    out_images = OUTPUT_DIR / "images"
    out_labels = OUTPUT_DIR / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    for orig, entries in sorted(index.items()):
        # Image: copy from the first entry (all entries share the same photo)
        src_img = entries[0][0]
        dst_img = out_images / (orig + src_img.suffix.lower())
        if not dst_img.exists():
            shutil.copy2(src_img, dst_img)

        # Labels: merge annotations from every source, remapping local → global id
        merged: list[str] = []
        for _, lbl_path, local_to_global in entries:
            for raw in lbl_path.read_text().splitlines():
                raw = raw.strip()
                if not raw:
                    continue
                parts = raw.split()
                local_id = int(parts[0])
                if local_id not in local_to_global:
                    print(f"  WARNING: unknown local class id {local_id} in '{lbl_path}' — skipping line")
                    continue
                parts[0] = str(local_to_global[local_id])
                merged.append(" ".join(parts))

        (out_labels / (orig + ".txt")).write_text("\n".join(merged) + "\n")

    # classes.txt  (ordered by global id)
    ordered_classes = [name for name, _ in sorted(global_class_id.items(), key=lambda x: x[1])]
    (OUTPUT_DIR / "classes.txt").write_text("\n".join(ordered_classes) + "\n")

    # notes.json
    notes = {
        "categories": [{"id": i, "name": name} for i, name in enumerate(ordered_classes)],
        "info": {
            "year": 2025,
            "version": "1.0",
            "contributor": "Label Studio"
        }
    }
    (OUTPUT_DIR / "notes.json").write_text(json.dumps(notes, indent=2) + "\n")

    print(f"\nDone!  Output → {OUTPUT_DIR}")
    print(f"  Images : {sum(1 for _ in out_images.iterdir())}")
    print(f"  Labels : {sum(1 for _ in out_labels.iterdir())}")
    print(f"  Classes: {ordered_classes}")


if __name__ == "__main__":
    main()