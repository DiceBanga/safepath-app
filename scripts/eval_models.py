"""Evaluate and compare SafePath segmentation models."""

# pyright: reportMissingImports=false

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sized, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.class_map import CLASS_MAP, HAZARD_IDS, NUM_CLASSES
from src.data.dataset import SafePathDataset, get_transforms
from src.models.deeplabv3plus import DeepLabV3Plus


IGNORE_INDEX = 255


@dataclass(frozen=True)
class EvalResult:
    """Serializable output schema for single-model evaluation."""

    model: str
    checkpoint: str
    model_size_mb: float
    metrics: Dict[str, Optional[float]]
    per_class_iou: Dict[str, float]
    num_parameters: int


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for single-model evaluation or comparison."""
    parser = argparse.ArgumentParser(description="Evaluate SafePath baseline/proposed segmentation models.")

    parser.add_argument("--compare", action="store_true", help="Evaluate baseline and proposed and write comparison markdown.")

    parser.add_argument(
        "--model",
        choices=("baseline", "proposed"),
        default=None,
        help="Single-model mode: model architecture to evaluate.",
    )
    parser.add_argument("--checkpoint", type=Path, default=None, help="Single-model mode: checkpoint path.")

    parser.add_argument("--baseline-checkpoint", type=Path, default=None, help="Comparison mode: baseline checkpoint path.")
    parser.add_argument("--proposed-checkpoint", type=Path, default=None, help="Comparison mode: proposed checkpoint path.")

    parser.add_argument("--data-dir", type=Path, required=True, help="Dataset root with images/<split> and masks/<split>.")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to evaluate (default: test).")
    parser.add_argument("--batch-size", type=int, default=4, help="Evaluation batch size.")
    parser.add_argument("--num-workers", type=int, default=4, help="Dataloader worker count.")
    parser.add_argument("--input-size", type=int, nargs=2, metavar=("HEIGHT", "WIDTH"), default=(480, 960), help="Evaluation input size.")
    parser.add_argument("--device", type=str, default=None, help="Device override (e.g., cpu, cuda).")

    parser.add_argument("--benchmark-fps", action="store_true", help="Benchmark inference FPS.")
    parser.add_argument("--num-warmup", type=int, default=10, help="FPS warmup iterations.")
    parser.add_argument("--num-iterations", type=int, default=100, help="FPS benchmark iterations.")

    parser.add_argument("--output", type=Path, required=True, help="Output file path (.json for single, .md for compare).")

    args = parser.parse_args()
    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    """Validate argument combinations and value ranges."""
    if args.compare:
        if args.model is not None or args.checkpoint is not None:
            raise ValueError("--compare cannot be combined with --model/--checkpoint")
        if args.baseline_checkpoint is None or args.proposed_checkpoint is None:
            raise ValueError("--compare requires --baseline-checkpoint and --proposed-checkpoint")
    else:
        if args.model is None or args.checkpoint is None:
            raise ValueError("Single-model mode requires --model and --checkpoint")

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")
    if args.num_warmup < 0:
        raise ValueError("--num-warmup cannot be negative")
    if args.num_iterations <= 0:
        raise ValueError("--num-iterations must be positive")
    if len(args.input_size) != 2 or args.input_size[0] <= 0 or args.input_size[1] <= 0:
        raise ValueError("--input-size must be two positive integers")


def resolve_device(device_arg: Optional[str]) -> torch.device:
    """Resolve compute device from user override or availability."""
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_eval_loader(
    data_dir: Path,
    split: str,
    batch_size: int,
    num_workers: int,
    input_size: Tuple[int, int],
) -> DataLoader:
    """Create evaluation dataloader."""
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    dataset = SafePathDataset(
        root_dir=str(data_dir),
        split=split,
        transform=get_transforms(mode="val", input_size=input_size),
        low_light_augment=False,
    )

    if len(dataset) == 0:
        raise ValueError(f"No samples found in split '{split}' at {data_dir}")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def create_model(model_name: str, device: torch.device) -> torch.nn.Module:
    """Instantiate supported SafePath model architecture."""
    if model_name == "baseline":
        from torchvision.models import segmentation as seg_models

        small_builder = getattr(seg_models, "lraspp_mobilenet_v3_small", None)
        large_builder = getattr(seg_models, "lraspp_mobilenet_v3_large", None)
        if small_builder is not None:
            lraspp_builder = small_builder
        elif large_builder is not None:
            lraspp_builder = large_builder
        else:
            raise RuntimeError("No LRASPP MobileNetV3 builder found in torchvision.models.segmentation")

        try:
            model = lraspp_builder(weights=None, weights_backbone=None, num_classes=NUM_CLASSES)
        except TypeError:
            model = lraspp_builder(weights=None, num_classes=NUM_CLASSES)
    elif model_name == "proposed":
        model = DeepLabV3Plus(num_classes=NUM_CLASSES, pretrained=False)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model.to(device)


def normalize_state_dict(state_dict: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
    """Normalize checkpoint state dict keys for loading."""
    normalized: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        key_str = str(key)
        if key_str.startswith("module."):
            key_str = key_str[len("module.") :]
        if isinstance(value, torch.Tensor):
            normalized[key_str] = value
    return normalized


def extract_state_dict(raw_checkpoint: Any) -> Dict[str, torch.Tensor]:
    """Extract model state dict from multiple checkpoint layouts."""
    if isinstance(raw_checkpoint, Mapping):
        if "model_state_dict" in raw_checkpoint and isinstance(raw_checkpoint["model_state_dict"], Mapping):
            return normalize_state_dict(raw_checkpoint["model_state_dict"])
        if "state_dict" in raw_checkpoint and isinstance(raw_checkpoint["state_dict"], Mapping):
            return normalize_state_dict(raw_checkpoint["state_dict"])
        if all(isinstance(v, torch.Tensor) for v in raw_checkpoint.values()):
            return normalize_state_dict(raw_checkpoint)
    raise ValueError("Unsupported checkpoint format; expected state_dict or model_state_dict")


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    """Load checkpoint weights into model."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    raw_checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = extract_state_dict(raw_checkpoint)
    model.load_state_dict(state_dict, strict=True)


def forward_logits(model: torch.nn.Module, images: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
    """Run forward pass and return resized logits."""
    outputs = model(images)
    logits = outputs["out"] if isinstance(outputs, Mapping) else outputs
    if logits.shape[-2:] != target_hw:
        logits = F.interpolate(logits, size=target_hw, mode="bilinear", align_corners=False)
    return logits


def update_confusion_matrix(
    confusion_matrix: torch.Tensor,
    prediction: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: int,
) -> torch.Tensor:
    """Accumulate confusion matrix with one batch."""
    valid_mask = target != ignore_index
    target_flat = target[valid_mask].reshape(-1)
    pred_flat = prediction[valid_mask].reshape(-1)

    if target_flat.numel() == 0:
        return confusion_matrix

    bin_indices = target_flat * num_classes + pred_flat
    batch_matrix = torch.bincount(bin_indices, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    confusion_matrix += batch_matrix.to(confusion_matrix.dtype)
    return confusion_matrix


def compute_iou_from_confusion(confusion_matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute per-class IoU and valid-class mask from confusion matrix."""
    true_positive = torch.diag(confusion_matrix)
    predicted = confusion_matrix.sum(dim=0)
    actual = confusion_matrix.sum(dim=1)
    union = actual + predicted - true_positive

    iou = torch.zeros_like(true_positive, dtype=torch.float64)
    valid = union > 0
    iou[valid] = true_positive[valid].double() / union[valid].double()
    return iou, valid


def compute_hazard_f1(confusion_matrix: torch.Tensor, hazard_ids: Iterable[int]) -> float:
    """Compute macro-F1 over hazard classes from confusion matrix."""
    f1_scores: List[float] = []
    predicted = confusion_matrix.sum(dim=0)
    actual = confusion_matrix.sum(dim=1)
    diagonal = torch.diag(confusion_matrix)

    for class_id in sorted(hazard_ids):
        tp = diagonal[class_id].double()
        fp = (predicted[class_id] - diagonal[class_id]).double()
        fn = (actual[class_id] - diagonal[class_id]).double()
        denominator = 2.0 * tp + fp + fn
        if denominator > 0:
            f1_scores.append(float((2.0 * tp / denominator).item()))
        else:
            f1_scores.append(0.0)

    return float(sum(f1_scores) / max(len(f1_scores), 1))


def evaluate_metrics(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, Dict[str, float], float]:
    """Run full evaluation and return mIoU, per-class IoU, hazard F1."""
    id_to_name = {class_id: class_name for class_name, class_id in CLASS_MAP.items()}

    confusion = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64, device=device)
    model.eval()

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.long().to(device, non_blocking=True)

            logits = forward_logits(model=model, images=images, target_hw=(masks.shape[-2], masks.shape[-1]))
            predictions = torch.argmax(logits, dim=1)
            confusion = update_confusion_matrix(
                confusion_matrix=confusion,
                prediction=predictions,
                target=masks,
                num_classes=NUM_CLASSES,
                ignore_index=IGNORE_INDEX,
            )

    iou_tensor, valid_mask = compute_iou_from_confusion(confusion)
    valid_iou = iou_tensor[valid_mask]
    miou = float(valid_iou.mean().item()) if valid_iou.numel() > 0 else 0.0

    per_class_iou = {id_to_name[idx]: float(iou_tensor[idx].item()) for idx in range(NUM_CLASSES)}
    hazard_f1 = compute_hazard_f1(confusion_matrix=confusion, hazard_ids=HAZARD_IDS)

    return miou, per_class_iou, hazard_f1


def synchronize_if_needed(device: torch.device) -> None:
    """Synchronize CUDA before/after timing when required."""
    if device.type == "cuda":
        torch.cuda.synchronize(device=device)


def benchmark_fps(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_warmup: int,
    num_iterations: int,
) -> float:
    """Measure inference throughput in FPS using a representative sample."""
    model.eval()

    first_batch = next(iter(loader))
    sample_images = first_batch[0].to(device, non_blocking=True)
    sample_target = first_batch[1]
    target_hw = (int(sample_target.shape[-2]), int(sample_target.shape[-1]))

    with torch.no_grad():
        for _ in range(num_warmup):
            _ = forward_logits(model=model, images=sample_images, target_hw=target_hw)

        synchronize_if_needed(device)
        start = time.perf_counter()
        for _ in range(num_iterations):
            _ = forward_logits(model=model, images=sample_images, target_hw=target_hw)
        synchronize_if_needed(device)
        elapsed = time.perf_counter() - start

    total_frames = num_iterations * int(sample_images.shape[0])
    return float(total_frames / max(elapsed, 1e-12))


def model_size_mb(checkpoint_path: Path) -> float:
    """Return checkpoint file size in megabytes."""
    return float(checkpoint_path.stat().st_size / (1024.0 * 1024.0))


def num_parameters(model: torch.nn.Module) -> int:
    """Count model parameters."""
    return int(sum(parameter.numel() for parameter in model.parameters()))


def run_single_evaluation(
    model_name: str,
    checkpoint_path: Path,
    loader: DataLoader,
    device: torch.device,
    benchmark: bool,
    num_warmup: int,
    num_iterations: int,
) -> EvalResult:
    """Evaluate one model and return all requested metrics."""
    model = create_model(model_name=model_name, device=device)
    load_checkpoint(model=model, checkpoint_path=checkpoint_path, device=device)

    miou, per_class_iou, hazard_f1 = evaluate_metrics(model=model, loader=loader, device=device)
    fps_value: Optional[float]
    if benchmark:
        fps_value = benchmark_fps(
            model=model,
            loader=loader,
            device=device,
            num_warmup=num_warmup,
            num_iterations=num_iterations,
        )
    else:
        fps_value = None

    descriptor = "baseline_mobilenetv3_small" if model_name == "baseline" else "proposed_deeplabv3plus_mobilenetv3_large"

    return EvalResult(
        model=descriptor,
        checkpoint=str(checkpoint_path),
        model_size_mb=model_size_mb(checkpoint_path),
        metrics={"mIoU": miou, "hazard_F1": hazard_f1, "fps": fps_value},
        per_class_iou=per_class_iou,
        num_parameters=num_parameters(model),
    )


def ensure_parent(path: Path) -> None:
    """Create parent directory for output if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)


def save_json(result: EvalResult, output_path: Path) -> None:
    """Save single-model evaluation output to JSON file."""
    ensure_parent(output_path)
    payload = {
        "model": result.model,
        "checkpoint": result.checkpoint,
        "model_size_mb": round(result.model_size_mb, 4),
        "metrics": {
            "mIoU": round(float(result.metrics["mIoU"] or 0.0), 6),
            "hazard_F1": round(float(result.metrics["hazard_F1"] or 0.0), 6),
            "fps": None if result.metrics["fps"] is None else round(float(result.metrics["fps"]), 6),
        },
        "per_class_iou": {name: round(value, 6) for name, value in result.per_class_iou.items()},
        "num_parameters": result.num_parameters,
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def metric_percent(value: Optional[float]) -> str:
    """Format metric value as percentage string."""
    if value is None:
        return "N/A"
    return f"{value * 100.0:.1f}%"


def metric_float(value: Optional[float], decimals: int = 2) -> str:
    """Format scalar metric with fixed precision."""
    if value is None:
        return "N/A"
    return f"{value:.{decimals}f}"


def format_parameters(num_params: int) -> str:
    """Format parameter count in millions."""
    return f"{num_params / 1_000_000.0:.2f}M"


def comparison_markdown(baseline: EvalResult, proposed: EvalResult) -> str:
    """Generate markdown table for side-by-side model comparison."""
    lines = [
        "| Metric | Baseline (MobileNetV3-Small) | Proposed (DeepLabV3+ MobileNetV3-Large) |",
        "|--------|------------------------------|----------------------------------------|",
        f"| mIoU | {metric_percent(baseline.metrics['mIoU'])} | {metric_percent(proposed.metrics['mIoU'])} |",
        f"| Hazard F1 | {metric_percent(baseline.metrics['hazard_F1'])} | {metric_percent(proposed.metrics['hazard_F1'])} |",
        f"| FPS | {metric_float(baseline.metrics['fps'])} | {metric_float(proposed.metrics['fps'])} |",
        f"| Model Size | {baseline.model_size_mb:.2f} MB | {proposed.model_size_mb:.2f} MB |",
        f"| Parameters | {format_parameters(baseline.num_parameters)} | {format_parameters(proposed.num_parameters)} |",
    ]
    return "\n".join(lines) + "\n"


def save_text(content: str, output_path: Path) -> None:
    """Save plain text output."""
    ensure_parent(output_path)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(content)


def print_single_summary(result: EvalResult) -> None:
    """Print human-readable summary for one model."""
    print(f"Model: {result.model}")
    print(f"Checkpoint: {result.checkpoint}")
    print(f"mIoU: {metric_percent(result.metrics['mIoU'])}")
    print(f"Hazard F1: {metric_percent(result.metrics['hazard_F1'])}")
    print(f"FPS: {metric_float(result.metrics['fps'])}")
    print(f"Model Size: {result.model_size_mb:.2f} MB")
    print(f"Parameters: {format_parameters(result.num_parameters)}")


def print_compare_summary(baseline: EvalResult, proposed: EvalResult, markdown_path: Path) -> None:
    """Print high-level comparison summary."""
    print("Comparison complete")
    print(f"Baseline mIoU: {metric_percent(baseline.metrics['mIoU'])}")
    print(f"Proposed mIoU: {metric_percent(proposed.metrics['mIoU'])}")
    print(f"Baseline Hazard F1: {metric_percent(baseline.metrics['hazard_F1'])}")
    print(f"Proposed Hazard F1: {metric_percent(proposed.metrics['hazard_F1'])}")
    print(f"Baseline FPS: {metric_float(baseline.metrics['fps'])}")
    print(f"Proposed FPS: {metric_float(proposed.metrics['fps'])}")
    print(f"Comparison markdown saved: {markdown_path}")


def main() -> int:
    """Entrypoint for model evaluation CLI."""
    args = parse_args()
    device = resolve_device(args.device)
    input_size = (int(args.input_size[0]), int(args.input_size[1]))
    loader = create_eval_loader(
        data_dir=args.data_dir,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        input_size=input_size,
    )

    dataset_obj: object = loader.dataset
    if isinstance(dataset_obj, Sized):
        eval_count = len(dataset_obj)
    else:
        eval_count = -1

    print(f"Using device: {device}")
    print(f"Eval samples: {eval_count} | split: {args.split}")

    if args.compare:
        baseline_result = run_single_evaluation(
            model_name="baseline",
            checkpoint_path=args.baseline_checkpoint,
            loader=loader,
            device=device,
            benchmark=args.benchmark_fps,
            num_warmup=args.num_warmup,
            num_iterations=args.num_iterations,
        )
        proposed_result = run_single_evaluation(
            model_name="proposed",
            checkpoint_path=args.proposed_checkpoint,
            loader=loader,
            device=device,
            benchmark=args.benchmark_fps,
            num_warmup=args.num_warmup,
            num_iterations=args.num_iterations,
        )

        markdown = comparison_markdown(baseline=baseline_result, proposed=proposed_result)
        save_text(markdown, args.output)

        comparison_json_path = args.output.with_suffix(".json")
        ensure_parent(comparison_json_path)
        with comparison_json_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "baseline": {
                        "model": baseline_result.model,
                        "checkpoint": baseline_result.checkpoint,
                        "model_size_mb": round(baseline_result.model_size_mb, 4),
                        "metrics": baseline_result.metrics,
                        "num_parameters": baseline_result.num_parameters,
                    },
                    "proposed": {
                        "model": proposed_result.model,
                        "checkpoint": proposed_result.checkpoint,
                        "model_size_mb": round(proposed_result.model_size_mb, 4),
                        "metrics": proposed_result.metrics,
                        "num_parameters": proposed_result.num_parameters,
                    },
                },
                handle,
                indent=2,
            )

        print_compare_summary(baseline=baseline_result, proposed=proposed_result, markdown_path=args.output)
        return 0

    single_result = run_single_evaluation(
        model_name=args.model,
        checkpoint_path=args.checkpoint,
        loader=loader,
        device=device,
        benchmark=args.benchmark_fps,
        num_warmup=args.num_warmup,
        num_iterations=args.num_iterations,
    )
    save_json(single_result, args.output)
    print_single_summary(single_result)
    print(f"JSON saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
