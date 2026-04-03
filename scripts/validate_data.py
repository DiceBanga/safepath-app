"""Validate SafePath segmentation datasets before training.

This script validates dataset integrity for one split or all splits under a
SafePath dataset root with the expected layout:

    data/<dataset_name>/
    ├── images/{train,val,test}/
    └── masks/{train,val,test}/

Validation checks:
1. 1:1 image/mask correspondence by file stem
2. image format support (PNG/JPG/JPEG)
3. mask label IDs are within [0, NUM_CLASSES - 1] or equal to 255 (ignore)
4. aggregate statistics and label distribution reporting

Exit codes:
    0 - validation succeeds
    1 - validation fails
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.class_map import CLASS_MAP, NUM_CLASSES


VALID_IMAGE_EXTENSIONS: Set[str] = {".png", ".jpg", ".jpeg"}
IGNORE_LABEL = 255
DEFAULT_SPLITS: Tuple[str, ...] = ("train", "val", "test")


@dataclass
class SplitStats:
    """Per-split validation statistics."""

    total_pairs: int = 0
    valid_pairs: int = 0
    invalid_pairs: int = 0
    warnings: int = 0


@dataclass
class ValidationSummary:
    """Aggregated validation output across one or more splits."""

    total_pairs: int = 0
    valid_pairs: int = 0
    invalid_pairs: int = 0
    warnings: int = 0
    label_distribution: Counter[int] = None  # type: ignore[assignment]
    split_stats: Dict[str, SplitStats] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.label_distribution is None:
            self.label_distribution = Counter()
        if self.split_stats is None:
            self.split_stats = defaultdict(SplitStats)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for dataset validation."""
    parser = argparse.ArgumentParser(
        description="Validate SafePath segmentation datasets before training."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Path to dataset root (contains images/ and masks/).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=("train", "val", "test", "all"),
        help="Dataset split to validate. Use 'all' for train/val/test.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as failures.",
    )
    return parser.parse_args()


def resolve_splits(split_arg: str) -> Sequence[str]:
    """Resolve the split argument into a concrete split list."""
    if split_arg == "all":
        return DEFAULT_SPLITS
    return (split_arg,)


def build_stem_index(directory: Path) -> Tuple[Dict[str, Path], List[str]]:
    """Build stem->path index and report duplicate stems in a directory."""
    stem_index: Dict[str, Path] = {}
    duplicates: List[str] = []

    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        stem = path.stem
        if stem in stem_index:
            duplicates.append(stem)
            continue
        stem_index[stem] = path

    return stem_index, duplicates


def is_supported_image_extension(path: Path) -> bool:
    """Return True when a file extension is a supported RGB image extension."""
    return path.suffix.lower() in VALID_IMAGE_EXTENSIONS


def can_open_image(path: Path) -> bool:
    """Check if a file can be opened by Pillow as an image."""
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (UnidentifiedImageError, OSError):
        return False


def load_mask_array(mask_path: Path) -> np.ndarray:
    """Load a segmentation mask into a numpy array."""
    with Image.open(mask_path) as mask_img:
        return np.array(mask_img)


def validate_mask_labels(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return unique mask IDs and invalid IDs detected in the mask."""
    if mask.ndim == 3:
        if mask.shape[2] == 1:
            mask = np.squeeze(mask, axis=2)
        else:
            flat_ids = np.unique(mask.reshape(-1))
            return flat_ids, flat_ids

    if mask.ndim != 2:
        flat_ids = np.unique(mask.reshape(-1))
        return flat_ids, flat_ids

    unique_ids = np.unique(mask)
    invalid_ids = unique_ids[(unique_ids != IGNORE_LABEL) & ((unique_ids < 0) | (unique_ids >= NUM_CLASSES))]
    return unique_ids, invalid_ids


def update_label_distribution(mask: np.ndarray, label_distribution: Counter[int]) -> None:
    """Accumulate per-label pixel counts from a mask."""
    unique_ids, counts = np.unique(mask, return_counts=True)
    for label_id, count in zip(unique_ids.tolist(), counts.tolist()):
        label_distribution[int(label_id)] += int(count)


def print_progress(current: int, total: int, split_name: str) -> None:
    """Print a simple textual progress indicator for split validation."""
    if total <= 0:
        return
    if current == total or current % max(1, total // 20) == 0:
        percent = (current / total) * 100.0
        print(f"  [{split_name}] validated {current}/{total} pairs ({percent:.1f}%)")


def validate_split(
    data_dir: Path,
    split_name: str,
    summary: ValidationSummary,
) -> None:
    """Validate all files for one dataset split and update summary in place."""
    images_dir = data_dir / "images" / split_name
    masks_dir = data_dir / "masks" / split_name

    split_stats = summary.split_stats[split_name]

    if not images_dir.is_dir() or not masks_dir.is_dir():
        print(
            f"[ERROR] Missing split directories for '{split_name}': "
            f"{images_dir} and/or {masks_dir}"
        )
        split_stats.invalid_pairs += 1
        summary.invalid_pairs += 1
        return

    image_index, image_duplicates = build_stem_index(images_dir)
    mask_index, mask_duplicates = build_stem_index(masks_dir)

    for stem in image_duplicates:
        print(f"[WARNING] Duplicate image stem '{stem}' in {images_dir}")
        split_stats.warnings += 1
        summary.warnings += 1

    for stem in mask_duplicates:
        print(f"[WARNING] Duplicate mask stem '{stem}' in {masks_dir}")
        split_stats.warnings += 1
        summary.warnings += 1

    all_stems = sorted(set(image_index.keys()) | set(mask_index.keys()))
    split_stats.total_pairs += len(all_stems)
    summary.total_pairs += len(all_stems)

    for idx, stem in enumerate(all_stems, start=1):
        image_path = image_index.get(stem)
        mask_path = mask_index.get(stem)

        pair_valid = True

        if image_path is None:
            print(f"[ERROR] Missing image for mask stem '{stem}' in split '{split_name}'")
            pair_valid = False

        if mask_path is None:
            print(f"[ERROR] Missing mask for image stem '{stem}' in split '{split_name}'")
            pair_valid = False

        if not pair_valid:
            split_stats.invalid_pairs += 1
            summary.invalid_pairs += 1
            print_progress(idx, len(all_stems), split_name)
            continue

        assert image_path is not None
        assert mask_path is not None

        if not is_supported_image_extension(image_path):
            print(
                f"[WARNING] Unsupported image extension for {image_path.name}. "
                f"Expected PNG/JPG."
            )
            split_stats.warnings += 1
            summary.warnings += 1
            pair_valid = False

        if not can_open_image(image_path):
            print(f"[ERROR] Cannot open image file: {image_path}")
            pair_valid = False

        if not can_open_image(mask_path):
            print(f"[ERROR] Cannot open mask file: {mask_path}")
            pair_valid = False

        if pair_valid:
            try:
                mask = load_mask_array(mask_path)
            except (UnidentifiedImageError, OSError, ValueError) as exc:
                print(f"[ERROR] Failed reading mask '{mask_path}': {exc}")
                pair_valid = False
            else:
                unique_ids, invalid_ids = validate_mask_labels(mask)
                if invalid_ids.size > 0:
                    invalid_str = ", ".join(str(int(x)) for x in invalid_ids.tolist())
                    print(
                        f"[ERROR] Invalid label IDs in mask '{mask_path.name}': "
                        f"{invalid_str}. Valid IDs are 0-{NUM_CLASSES - 1} and {IGNORE_LABEL}."
                    )
                    pair_valid = False
                else:
                    update_label_distribution(mask, summary.label_distribution)

                if unique_ids.size == 1 and int(unique_ids[0]) == IGNORE_LABEL:
                    print(f"[WARNING] Mask '{mask_path.name}' contains only ignore label ({IGNORE_LABEL}).")
                    split_stats.warnings += 1
                    summary.warnings += 1

        if pair_valid:
            split_stats.valid_pairs += 1
            summary.valid_pairs += 1
        else:
            split_stats.invalid_pairs += 1
            summary.invalid_pairs += 1

        print_progress(idx, len(all_stems), split_name)


def print_label_distribution(label_distribution: Counter[int]) -> None:
    """Print sorted label distribution with class names where available."""
    print("\nLabel Distribution (pixel counts):")
    if not label_distribution:
        print("  (no valid mask labels found)")
        return

    id_to_name = {class_id: class_name for class_name, class_id in CLASS_MAP.items()}

    for label_id in sorted(label_distribution.keys()):
        if label_id == IGNORE_LABEL:
            label_name = "ignore"
        else:
            label_name = id_to_name.get(label_id, "unknown")
        print(f"  {label_id:>3} ({label_name:<15}): {label_distribution[label_id]}")


def print_summary(summary: ValidationSummary) -> None:
    """Print split-level and global validation summary."""
    print("\nValidation Summary")
    print("-" * 72)
    for split_name in sorted(summary.split_stats.keys()):
        split = summary.split_stats[split_name]
        print(
            f"Split '{split_name}': total={split.total_pairs}, "
            f"valid={split.valid_pairs}, invalid={split.invalid_pairs}, warnings={split.warnings}"
        )

    print("-" * 72)
    print(f"Total pairs:   {summary.total_pairs}")
    print(f"Valid pairs:   {summary.valid_pairs}")
    print(f"Invalid pairs: {summary.invalid_pairs}")
    print(f"Warnings:      {summary.warnings}")
    print_label_distribution(summary.label_distribution)


def validate_dataset(data_dir: Path, splits: Sequence[str]) -> ValidationSummary:
    """Validate selected dataset splits and return aggregate results."""
    summary = ValidationSummary()
    for split_name in splits:
        print(f"\nValidating split: {split_name}")
        validate_split(data_dir, split_name, summary)
    return summary


def main() -> int:
    """CLI entrypoint for dataset validation."""
    args = parse_args()
    data_dir = args.data_dir

    if not data_dir.is_dir():
        print(f"[ERROR] Data directory does not exist: {data_dir}")
        return 1

    splits = resolve_splits(args.split)

    images_root = data_dir / "images"
    masks_root = data_dir / "masks"
    if not images_root.is_dir() or not masks_root.is_dir():
        print(
            f"[ERROR] Expected dataset directories missing under {data_dir}: "
            f"required 'images/' and 'masks/'"
        )
        return 1

    summary = validate_dataset(data_dir, splits)
    print_summary(summary)

    failed = summary.invalid_pairs > 0 or (args.strict and summary.warnings > 0)
    if failed:
        if args.strict and summary.warnings > 0 and summary.invalid_pairs == 0:
            print("\n[RESULT] Validation failed due to strict mode warnings.")
        else:
            print("\n[RESULT] Validation failed.")
        return 1

    print("\n[RESULT] Validation succeeded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
