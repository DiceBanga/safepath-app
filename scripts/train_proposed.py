"""Train proposed DeepLabV3+ model using baseline training pipeline."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.train_baseline as baseline
from src.data.class_map import NUM_CLASSES
from src.models.deeplabv3plus import DeepLabV3Plus


def create_model(device: Any) -> Any:
    """Create DeepLabV3+ proposed model for SafePath classes."""
    model = DeepLabV3Plus(num_classes=NUM_CLASSES, pretrained=True)
    return model.to(device)


baseline.create_model = create_model


if __name__ == "__main__":
    raise SystemExit(baseline.main())
