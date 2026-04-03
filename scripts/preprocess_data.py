"""Preprocess raw segmentation datasets into SafePath format.

This script converts Cityscapes or BDD100K raw segmentation layouts into the
unified SafePath directory structure:

    <output>/
    ├── images/<split>/
    └── masks/<split>/

For each selected split, the script:
1. Discovers image/mask pairs from the source dataset layout.
2. Loads images and masks with Pillow.
3. Optionally resizes image/mask pairs to a target (height, width).
4. Remaps source mask IDs to SafePath IDs via ``src.data.class_map`` helpers.
5. Writes copied images and remapped masks to the output dataset tree.
6. Optionally creates low-light augmented training images.

Notes:
- Original datasets are never modified.
- Masks are always saved as PNG with SafePath IDs.
- Dry-run mode reports actions without writing files.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.class_map import NUM_CLASSES, remap_bdd100k, remap_cityscapes


CITYSCAPES = "cityscapes"
BDD100K = "bdd100k"
SUPPORTED_DATASETS = (CITYSCAPES, BDD100K)
VALID_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
IGNORE_LABEL = 255


BDD100K_DEFAULT_LABEL_MAP: Dict[str, int] = {
    "road": 0,
    "sidewalk": 1,
    "building": 2,
    "wall": 3,
    "fence": 4,
    "pole": 5,
    "traffic_light": 6,
    "traffic_sign": 7,
    "vegetation": 8,
    "terrain": 9,
    "sky": 10,
    "person": 11,
    "rider": 12,
    "car": 13,
    "truck": 14,
    "bus": 15,
    "train": 16,
    "motorcycle": 17,
    "bicycle": 18,
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for dataset preprocessing."""
    parser = argparse.ArgumentParser(
        description="Convert Cityscapes/BDD100K into SafePath segmentation format."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=SUPPORTED_DATASETS,
        help="Source dataset to preprocess.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to raw dataset root.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to output SafePath dataset root.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=("train", "val"),
        help="Dataset splits to process (default: train val).",
    )
    parser.add_argument(
        "--resize",
        nargs=2,
        type=int,
        metavar=("HEIGHT", "WIDTH"),
        help="Optional target resolution to resize pairs before writing.",
    )
    parser.add_argument(
        "--low-light-augment",
        action="store_true",
        help="Create additional low-light images for the training split.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview actions without writing output files.",
    )
    return parser.parse_args()


def normalize_splits(raw_splits: Sequence[str]) -> List[str]:
    """Return normalized split names while preserving order and uniqueness."""
    normalized: List[str] = []
    seen = set()
    for split in raw_splits:
        split_name = split.strip().lower()
        if not split_name:
            continue
        if split_name in seen:
            continue
        seen.add(split_name)
        normalized.append(split_name)
    return normalized


def validate_resize(resize: Optional[Sequence[int]]) -> Optional[Tuple[int, int]]:
    """Validate and normalize resize arguments to ``(height, width)``."""
    if resize is None:
        return None

    if len(resize) != 2:
        raise ValueError("Resize must provide exactly two integers: HEIGHT WIDTH")

    height = int(resize[0])
    width = int(resize[1])
    if height <= 0 or width <= 0:
        raise ValueError("Resize dimensions must be positive integers")
    return height, width


def collect_cityscapes_pairs(input_root: Path, split: str) -> Tuple[List[Tuple[Path, Path, str]], int]:
    """Collect valid Cityscapes ``(image_path, mask_path, output_stem)`` pairs."""
    images_root = input_root / "leftImg8bit" / split
    masks_root = input_root / "gtFine" / split

    if not images_root.is_dir() or not masks_root.is_dir():
        print(
            f"[ERROR] Missing Cityscapes split directories for '{split}': "
            f"{images_root} and/or {masks_root}"
        )
        return [], 0

    pairs: List[Tuple[Path, Path, str]] = []
    missing_masks = 0
    for image_path in sorted(images_root.rglob("*_leftImg8bit.png")):
        city_name = image_path.parent.name
        suffix = "_leftImg8bit.png"
        if not image_path.name.endswith(suffix):
            continue

        base = image_path.name[: -len(suffix)]
        mask_path = masks_root / city_name / f"{base}_gtFine_labelIds.png"
        if not mask_path.is_file():
            print(f"[WARNING] Missing mask for image '{image_path.name}': {mask_path}")
            missing_masks += 1
            continue

        output_stem = f"{city_name}_{base}"
        pairs.append((image_path, mask_path, output_stem))

    return pairs, missing_masks


def collect_bdd100k_pairs(input_root: Path, split: str) -> Tuple[List[Tuple[Path, Path, str]], int]:
    """Collect valid BDD100K ``(image_path, mask_path, output_stem)`` pairs."""
    images_root = input_root / "images" / "10k" / split
    masks_root = input_root / "labels" / "sem_seg" / "masks" / split

    if not images_root.is_dir() or not masks_root.is_dir():
        print(
            f"[ERROR] Missing BDD100K split directories for '{split}': "
            f"{images_root} and/or {masks_root}"
        )
        return [], 0

    pairs: List[Tuple[Path, Path, str]] = []
    missing_masks = 0
    for image_path in sorted(images_root.iterdir()):
        if not image_path.is_file():
            continue
        if image_path.suffix.lower() not in VALID_IMAGE_EXTENSIONS:
            continue

        output_stem = image_path.stem
        mask_path = masks_root / f"{output_stem}_train_id.png"
        if not mask_path.is_file():
            print(f"[WARNING] Missing mask for image '{image_path.name}': {mask_path}")
            missing_masks += 1
            continue
        pairs.append((image_path, mask_path, output_stem))

    return pairs, missing_masks


def build_pairs(dataset: str, input_root: Path, split: str) -> Tuple[List[Tuple[Path, Path, str]], int]:
    """Dispatch pair collection based on dataset name."""
    if dataset == CITYSCAPES:
        return collect_cityscapes_pairs(input_root, split)
    if dataset == BDD100K:
        return collect_bdd100k_pairs(input_root, split)
    raise ValueError(f"Unsupported dataset '{dataset}'")


def ensure_output_dirs(output_root: Path, split: str, dry_run: bool) -> Tuple[Path, Path]:
    """Create output image/mask split directories unless in dry-run mode."""
    images_dir = output_root / "images" / split
    masks_dir = output_root / "masks" / split
    if not dry_run:
        images_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)
    return images_dir, masks_dir


def load_rgb_image(image_path: Path) -> Image.Image:
    """Load an image path as RGB."""
    with Image.open(image_path) as image:
        return image.convert("RGB")


def load_mask_array(mask_path: Path) -> np.ndarray:
    """Load a segmentation mask as raw numeric IDs from image data."""
    with Image.open(mask_path) as mask:
        mask_array = np.array(mask)
    if mask_array.ndim == 3:
        mask_array = mask_array[:, :, 0]
    return mask_array


def maybe_resize_pair(
    image: Image.Image,
    mask: Image.Image,
    resize: Optional[Tuple[int, int]],
) -> Tuple[Image.Image, Image.Image]:
    """Resize image/mask pair to ``(height, width)`` if requested."""
    if resize is None:
        return image, mask

    height, width = resize
    resized_image = image.resize((width, height), resample=Image.Resampling.BILINEAR)
    resized_mask = mask.resize((width, height), resample=Image.Resampling.NEAREST)
    return resized_image, resized_mask


def remap_mask(dataset: str, mask_array: np.ndarray) -> Image.Image:
    """Remap raw dataset mask IDs to SafePath IDs and return a PIL mask."""
    if dataset == CITYSCAPES:
        remapped = remap_cityscapes(mask_array)
    elif dataset == BDD100K:
        remapped = remap_bdd100k(mask_array, BDD100K_DEFAULT_LABEL_MAP)
    else:
        raise ValueError(f"Unsupported dataset '{dataset}'")

    remapped = remapped.astype(np.uint8, copy=False)
    invalid = np.unique(remapped[(remapped != IGNORE_LABEL) & (remapped >= NUM_CLASSES)])
    if invalid.size > 0:
        invalid_values = ", ".join(str(int(v)) for v in invalid.tolist())
        raise ValueError(
            f"Remapped mask contains invalid IDs: {invalid_values}. "
            f"Expected 0-{NUM_CLASSES - 1} or {IGNORE_LABEL}."
        )
    return Image.fromarray(remapped, mode="L")


def create_low_light_image(image: Image.Image) -> Image.Image:
    """Create a simple low-light variant using multiplicative darkening + gamma."""
    image_array = np.asarray(image).astype(np.float32) / 255.0
    image_array = np.clip(image_array * 0.60, 0.0, 1.0)
    image_array = np.power(image_array, 1.35)
    image_array = np.clip(np.round(image_array * 255.0), 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(image_array, mode="RGB")


def save_image(image: Image.Image, output_path: Path, dry_run: bool) -> None:
    """Save an RGB image to disk, preserving extension semantics."""
    if dry_run:
        print(f"[DRY-RUN] Would write image: {output_path}")
        return

    suffix = output_path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        image.save(output_path, quality=95)
    else:
        image.save(output_path)


def save_mask(mask: Image.Image, output_path: Path, dry_run: bool) -> None:
    """Save a remapped mask as PNG."""
    if dry_run:
        print(f"[DRY-RUN] Would write mask:  {output_path}")
        return
    mask.save(output_path)


def print_progress(current: int, total: int, split: str) -> None:
    """Print preprocessing progress for one split."""
    if total <= 0:
        return
    print(f"[{split}] Processed {current}/{total} images")


def preprocess_split(
    dataset: str,
    input_root: Path,
    output_root: Path,
    split: str,
    resize: Optional[Tuple[int, int]],
    low_light_augment: bool,
    dry_run: bool,
) -> Dict[str, int]:
    """Preprocess one split and return counters for summary reporting."""
    pairs, missing_masks = build_pairs(dataset, input_root, split)
    images_dir, masks_dir = ensure_output_dirs(output_root, split, dry_run)

    stats = {
        "discovered": len(pairs),
        "processed": 0,
        "failed": 0,
        "missing_masks": missing_masks,
        "augmented": 0,
    }

    if not pairs:
        print(f"[INFO] No valid pairs found for split '{split}'.")
        return stats

    total = len(pairs)
    for idx, (image_path, mask_path, output_stem) in enumerate(pairs, start=1):
        try:
            image = load_rgb_image(image_path)
            mask_array = load_mask_array(mask_path)
            mask = Image.fromarray(mask_array.astype(np.uint8), mode="L")
            image, mask = maybe_resize_pair(image, mask, resize)
            remapped_mask = remap_mask(dataset, np.array(mask))

            output_image_path = images_dir / f"{output_stem}{image_path.suffix.lower()}"
            output_mask_path = masks_dir / f"{output_stem}.png"
            save_image(image, output_image_path, dry_run)
            save_mask(remapped_mask, output_mask_path, dry_run)
            stats["processed"] += 1

            if low_light_augment and split == "train":
                augmented_image = create_low_light_image(image)
                aug_stem = f"{output_stem}_lowlight"
                save_image(augmented_image, images_dir / f"{aug_stem}{image_path.suffix.lower()}", dry_run)
                save_mask(remapped_mask, masks_dir / f"{aug_stem}.png", dry_run)
                stats["augmented"] += 1

        except (UnidentifiedImageError, OSError, ValueError) as exc:
            print(
                f"[ERROR] Failed processing pair '\n"
                f"        image={image_path}\n"
                f"        mask={mask_path}\n"
                f"        reason={exc}"
            )
            stats["failed"] += 1

        print_progress(idx, total, split)

    return stats


def print_summary(
    dataset: str,
    output_root: Path,
    split_stats: Dict[str, Dict[str, int]],
    dry_run: bool,
) -> None:
    """Print end-of-run summary information."""
    total_discovered = sum(stats["discovered"] for stats in split_stats.values())
    total_processed = sum(stats["processed"] for stats in split_stats.values())
    total_failed = sum(stats["failed"] for stats in split_stats.values())
    total_missing = sum(stats["missing_masks"] for stats in split_stats.values())
    total_augmented = sum(stats["augmented"] for stats in split_stats.values())

    print("\nPreprocessing Summary")
    print("-" * 72)
    print(f"Dataset:            {dataset}")
    print(f"Total discovered:   {total_discovered}")
    print(f"Total processed:    {total_processed}")
    print(f"Total failed:       {total_failed}")
    print(f"Missing masks:      {total_missing}")
    print(f"Low-light outputs:  {total_augmented}")

    for split, stats in split_stats.items():
        print(
            f"Split '{split}': discovered={stats['discovered']}, "
            f"processed={stats['processed']}, failed={stats['failed']}, "
            f"missing_masks={stats['missing_masks']}, augmented={stats['augmented']}"
        )

    if dry_run:
        print(f"\n[DRY-RUN] No files were written. Output preview root: {output_root}")
    else:
        print(f"\nOutput written to: {output_root}")


def preprocess_dataset(args: argparse.Namespace) -> int:
    """Drive preprocessing across selected splits and return exit status."""
    input_root = args.input
    output_root = args.output
    splits = normalize_splits(args.splits)

    if not input_root.is_dir():
        print(f"[ERROR] Input directory does not exist: {input_root}")
        return 1

    if not splits:
        print("[ERROR] No valid splits provided.")
        return 1

    try:
        resize = validate_resize(args.resize)
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        return 1

    split_stats: Dict[str, Dict[str, int]] = {}
    for split in splits:
        print(f"\nProcessing split: {split}")
        split_stats[split] = preprocess_split(
            dataset=args.dataset,
            input_root=input_root,
            output_root=output_root,
            split=split,
            resize=resize,
            low_light_augment=args.low_light_augment,
            dry_run=args.dry_run,
        )

    print_summary(args.dataset, output_root, split_stats, args.dry_run)

    total_processed = sum(stats["processed"] for stats in split_stats.values())
    total_failed = sum(stats["failed"] for stats in split_stats.values())
    if total_processed == 0 and total_failed == 0:
        print("\n[RESULT] No data processed.")
        return 1
    if total_failed > 0:
        print("\n[RESULT] Completed with errors.")
        return 1
    print("\n[RESULT] Completed successfully.")
    return 0


def main() -> int:
    """CLI entrypoint."""
    args = parse_args()
    return preprocess_dataset(args)


if __name__ == "__main__":
    raise SystemExit(main())
