"""Export SafePath models to ONNX format for mobile deployment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from src.data.class_map import NUM_CLASSES


def load_baseline(device: torch.device) -> Any:
    from torchvision.models import segmentation as seg_models

    small_builder = getattr(seg_models, "lraspp_mobilenet_v3_small", None)
    large_builder = getattr(seg_models, "lraspp_mobilenet_v3_large", None)
    if small_builder is not None:
        lraspp_builder = small_builder
    elif large_builder is not None:
        lraspp_builder = large_builder
    else:
        raise RuntimeError("No LRASPP MobileNetV3 builder found")

    try:
        model = lraspp_builder(weights=None, weights_backbone=None, num_classes=NUM_CLASSES)
    except TypeError:
        model = lraspp_builder(num_classes=NUM_CLASSES)

    return model.to(device)


def load_proposed(device: torch.device) -> Any:
    from src.models.deeplabv3plus import DeepLabV3Plus

    model = DeepLabV3Plus(num_classes=NUM_CLASSES, pretrained=False)
    return model.to(device)


def export_model(
    model: torch.nn.Module,
    output_path: Path,
    input_size: tuple[int, int] = (128, 256),
    opset_version: int = 14
) -> None:
    model.eval()
    dummy_input = torch.randn(1, 3, *input_size)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        },
        opset_version=opset_version
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["baseline", "proposed"])
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--input-size", type=int, nargs=2, default=[128, 256])
    args = parser.parse_args()

    device = torch.device("cpu")
    if args.model == "baseline":
        model = load_baseline(device)
    else:
        model = load_proposed(device)
    model.to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    export_model(model, args.output, tuple(args.input_size))
    print(f"Exported {args.model} to {args.output}")


if __name__ == "__main__":
    raise SystemExit(main())
