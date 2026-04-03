"""Unit tests for scripts/preprocess_data.py helpers."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.preprocess_data import (
    CITYSCAPES,
    BDD100K,
    BDD100K_DEFAULT_LABEL_MAP,
    normalize_splits,
    validate_resize,
    maybe_resize_pair,
    remap_mask,
    create_low_light_image,
    collect_cityscapes_pairs,
    collect_bdd100k_pairs,
)


class TestNormalizeSplits:
    def test_basic_normalization(self) -> None:
        result = normalize_splits(["TRAIN", "Val", "TEST"])
        assert result == ["train", "val", "test"]

    def test_deduplication(self) -> None:
        result = normalize_splits(["train", "TRAIN", "val", "  val  "])
        assert result == ["train", "val"]

    def test_empty_filtering(self) -> None:
        result = normalize_splits(["train", "", "  ", "val"])
        assert result == ["train", "val"]

    def test_empty_input(self) -> None:
        assert normalize_splits([]) == []


class TestValidateResize:
    def test_none_returns_none(self) -> None:
        assert validate_resize(None) is None

    def test_valid_resize(self) -> None:
        assert validate_resize([480, 960]) == (480, 960)

    def test_invalid_length(self) -> None:
        with pytest.raises(ValueError, match="exactly two integers"):
            validate_resize([480])

    def test_negative_value(self) -> None:
        with pytest.raises(ValueError, match="positive integers"):
            validate_resize([-10, 960])

    def test_zero_value(self) -> None:
        with pytest.raises(ValueError, match="positive integers"):
            validate_resize([480, 0])


class TestMaybeResizePair:
    def _create_test_image(self, size: Tuple[int, int]) -> Image.Image:
        arr = np.random.randint(0, 256, (*size, 3), dtype=np.uint8)
        return Image.fromarray(arr, mode="RGB")

    def _create_test_mask(self, size: Tuple[int, int]) -> Image.Image:
        arr = np.random.randint(0, 23, size, dtype=np.uint8)
        return Image.fromarray(arr, mode="L")

    def test_no_resize_returns_same(self) -> None:
        image = self._create_test_image((100, 200))
        mask = self._create_test_mask((100, 200))
        out_image, out_mask = maybe_resize_pair(image, mask, None)
        assert out_image.size == image.size
        assert out_mask.size == mask.size

    def test_resize_changes_dimensions(self) -> None:
        image = self._create_test_image((100, 200))
        mask = self._create_test_mask((100, 200))
        out_image, out_mask = maybe_resize_pair(image, mask, (50, 100))
        assert out_image.size == (100, 50)
        assert out_mask.size == (100, 50)


class TestRemapMask:
    def test_cityscapes_basic_remap(self) -> None:
        mask_array = np.array([[0, 1], [11, 255]], dtype=np.uint8)
        mask_img = Image.fromarray(mask_array, mode="L")
        remapped = remap_mask(CITYSCAPES, np.array(mask_img))
        result = np.array(remapped)
        assert result[0, 0] == 0
        assert result[0, 1] == 1
        assert result[1, 0] == 11
        assert result[1, 1] == 255

    def test_bdd100k_basic_remap(self) -> None:
        mask_array = np.array([[0, 1], [11, 13]], dtype=np.uint8)
        mask_img = Image.fromarray(mask_array, mode="L")
        remapped = remap_mask(BDD100K, np.array(mask_img))
        result = np.array(remapped)
        assert result[0, 0] == 0
        assert result[0, 1] == 1
        assert result[1, 0] == 11
        assert result[1, 1] == 13

    def test_invalid_dataset_raises(self) -> None:
        mask_array = np.zeros((10, 10), dtype=np.uint8)
        with pytest.raises(ValueError, match="Unsupported dataset"):
            remap_mask("unknown", mask_array)


class TestCreateLowLightImage:
    def test_output_is_rgb(self) -> None:
        arr = np.full((10, 10, 3), 200, dtype=np.uint8)
        image = Image.fromarray(arr, mode="RGB")
        result = create_low_light_image(image)
        assert result.mode == "RGB"
        assert result.size == image.size

    def test_output_is_darker(self) -> None:
        arr = np.full((10, 10, 3), 200, dtype=np.uint8)
        image = Image.fromarray(arr, mode="RGB")
        result = create_low_light_image(image)
        result_arr = np.array(result)
        original_mean = np.mean(arr)
        result_mean = np.mean(result_arr)
        assert result_mean < original_mean


class TestCollectCityscapesPairs:
    def test_missing_directories_returns_empty(self, tmp_path: Path) -> None:
        pairs, missing = collect_cityscapes_pairs(tmp_path, "train")
        assert pairs == []
        assert missing == 0


class TestCollectBDD100KPairs:
    def test_missing_directories_returns_empty(self, tmp_path: Path) -> None:
        pairs, missing = collect_bdd100k_pairs(tmp_path, "train")
        assert pairs == []
        assert missing == 0

    def test_finds_valid_pairs(self, tmp_path: Path) -> None:
        images_dir = tmp_path / "images" / "10k" / "train"
        masks_dir = tmp_path / "labels" / "sem_seg" / "masks" / "train"
        images_dir.mkdir(parents=True)
        masks_dir.mkdir(parents=True)

        image_path = images_dir / "sample.jpg"
        Image.new("RGB", (64, 64)).save(image_path)

        mask_path = masks_dir / "sample.png"
        Image.new("L", (64, 64)).save(mask_path)

        pairs, missing = collect_bdd100k_pairs(tmp_path, "train")
        assert len(pairs) == 1
        assert missing == 0
        assert pairs[0][2] == "sample"

    def test_reports_missing_mask(self, tmp_path: Path) -> None:
        images_dir = tmp_path / "images" / "10k" / "train"
        masks_dir = tmp_path / "labels" / "sem_seg" / "masks" / "train"
        images_dir.mkdir(parents=True)
        masks_dir.mkdir(parents=True)

        image_path = images_dir / "sample.jpg"
        Image.new("RGB", (64, 64)).save(image_path)

        pairs, missing = collect_bdd100k_pairs(tmp_path, "train")
        assert pairs == []
        assert missing == 1
