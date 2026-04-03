"""Train LRASPP MobileNetV3-Small baseline for SafePath segmentation."""

# pyright: reportMissingImports=false

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.class_map import NUM_CLASSES


IGNORE_INDEX = 255


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for baseline training."""
    parser = argparse.ArgumentParser(
        description="Train LRASPP MobileNetV3-Small baseline on SafePath dataset."
    )
    parser.add_argument("--config", type=Path, default=None, help="Optional YAML config file.")
    parser.add_argument("--data-dir", type=Path, default=None, help="SafePath dataset root directory.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory.")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate.")
    parser.add_argument("--num-workers", type=int, default=None, help="Loader workers.")
    parser.add_argument("--resume", type=Path, default=None, help="Checkpoint path to resume.")
    parser.add_argument("--weight-decay", type=float, default=None, help="Adam weight decay.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def load_yaml_config(path: Optional[Path]) -> Dict[str, Any]:
    """Load YAML config dictionary."""
    if path is None:
        return {}
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("Config file must parse to a dictionary.")
    return data


def resolve_config(args: argparse.Namespace, config: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve runtime config from YAML defaults and CLI overrides."""
    training_cfg = config.get("training", {}) if isinstance(config.get("training"), dict) else {}
    data_cfg = config.get("data", {}) if isinstance(config.get("data"), dict) else {}

    data_dir = args.data_dir or Path(config.get("data_dir", "data/bdd100k_safepath"))
    output_dir = args.output_dir or Path(config.get("output_dir", "models/baseline_mobilenetv3_small"))
    epochs = args.epochs if args.epochs is not None else int(training_cfg.get("num_epochs", 50))
    batch_size = args.batch_size if args.batch_size is not None else int(training_cfg.get("batch_size", 8))
    lr = args.lr if args.lr is not None else float(training_cfg.get("learning_rate", 1e-3))
    num_workers = args.num_workers if args.num_workers is not None else int(data_cfg.get("num_workers", 4))
    weight_decay = (
        args.weight_decay if args.weight_decay is not None else float(training_cfg.get("weight_decay", 1e-4))
    )

    input_size_raw: Sequence[int] = data_cfg.get("input_size", [480, 960])
    if len(input_size_raw) != 2:
        raise ValueError("config.data.input_size must be [height, width]")
    input_size = (int(input_size_raw[0]), int(input_size_raw[1]))

    resolved = {
        "data_dir": Path(data_dir),
        "output_dir": Path(output_dir),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "num_workers": int(num_workers),
        "resume": args.resume,
        "weight_decay": float(weight_decay),
        "seed": int(args.seed),
        "input_size": input_size,
    }

    if resolved["epochs"] <= 0:
        raise ValueError("epochs must be positive")
    if resolved["batch_size"] <= 0:
        raise ValueError("batch-size must be positive")
    if resolved["num_workers"] < 0:
        raise ValueError("num-workers cannot be negative")
    if resolved["lr"] <= 0:
        raise ValueError("lr must be positive")

    return resolved


def set_seed(seed: int) -> None:
    """Set torch seeds for reproducibility."""
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_dataloaders(
    data_dir: Path,
    batch_size: int,
    num_workers: int,
    input_size: Tuple[int, int],
) -> Tuple[Any, Any]:
    """Create train/validation data loaders."""
    import torch
    from torch.utils.data import DataLoader
    from src.data.dataset import SafePathDataset, get_transforms

    train_dataset = SafePathDataset(
        root_dir=str(data_dir),
        split="train",
        transform=get_transforms(mode="train", input_size=input_size),
        low_light_augment=False,
    )
    val_dataset = SafePathDataset(
        root_dir=str(data_dir),
        split="val",
        transform=get_transforms(mode="val", input_size=input_size),
        low_light_augment=False,
    )

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader


def create_model(device: Any) -> Any:
    """Create LRASPP MobileNetV3 baseline model with 23 SafePath classes."""
    from torchvision.models import segmentation as seg_models

    small_builder = getattr(seg_models, "lraspp_mobilenet_v3_small", None)
    large_builder = getattr(seg_models, "lraspp_mobilenet_v3_large", None)
    if small_builder is not None:
        lraspp_builder = small_builder
        weight_enum_name = "LRASPP_MobileNet_V3_Small_Weights"
    elif large_builder is not None:
        lraspp_builder = large_builder
        weight_enum_name = "LRASPP_MobileNet_V3_Large_Weights"
    else:
        raise RuntimeError("No LRASPP MobileNetV3 builder found in torchvision.models.segmentation")

    try:
        weights_enum = getattr(seg_models, weight_enum_name)
        model = lraspp_builder(weights=weights_enum.DEFAULT, num_classes=NUM_CLASSES)
    except Exception:
        try:
            model = lraspp_builder(weights=None, weights_backbone=None, num_classes=NUM_CLASSES)
        except TypeError:
            model = lraspp_builder(num_classes=NUM_CLASSES)
    return model.to(device)


def compute_batch_intersection_union(
    pred: Any,
    target: Any,
    num_classes: int,
    ignore_index: int,
) -> Tuple[Any, Any]:
    """Compute per-class intersection and union tensors for one batch."""
    import torch

    valid_mask = target != ignore_index
    pred = pred[valid_mask]
    target = target[valid_mask]

    if target.numel() == 0:
        zeros = torch.zeros(num_classes, dtype=torch.float64, device=target.device)
        return zeros, zeros

    intersection = torch.zeros(num_classes, dtype=torch.float64, device=target.device)
    union = torch.zeros(num_classes, dtype=torch.float64, device=target.device)
    for class_idx in range(num_classes):
        pred_mask = pred == class_idx
        target_mask = target == class_idx
        intersection[class_idx] = torch.sum(pred_mask & target_mask).double()
        union[class_idx] = torch.sum(pred_mask | target_mask).double()

    return intersection, union


def forward_logits(model: Any, images: Any, target_shape: Tuple[int, int]) -> Any:
    """Run model and return logits resized to target shape."""
    import torch.nn.functional as F

    outputs = model(images)
    logits = outputs["out"] if isinstance(outputs, dict) else outputs
    if logits.shape[-2:] != target_shape:
        logits = F.interpolate(logits, size=target_shape, mode="bilinear", align_corners=False)
    return logits


def train_one_epoch(
    model: Any,
    loader: Any,
    criterion: Any,
    optimizer: Any,
    device: Any,
    writer: Any,
    epoch: int,
    global_step: int,
) -> Tuple[float, int]:
    """Train for one epoch and return epoch loss + updated global step."""
    from tqdm import tqdm

    model.train()
    running_loss = 0.0

    for images, masks in tqdm(loader, desc=f"Train {epoch + 1}", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.long().to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = forward_logits(model, images, target_shape=(masks.shape[-2], masks.shape[-1]))
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        loss_value = float(loss.item())
        running_loss += loss_value
        writer.add_scalar("train/step_loss", loss_value, global_step)
        global_step += 1

    avg_loss = running_loss / max(len(loader), 1)
    writer.add_scalar("train/epoch_loss", avg_loss, epoch)
    return avg_loss, global_step


def validate(model: Any, loader: Any, criterion: Any, device: Any) -> Tuple[float, float]:
    """Run validation and return (loss, mIoU)."""
    import torch
    from tqdm import tqdm

    model.eval()
    running_loss = 0.0
    total_intersection = torch.zeros(NUM_CLASSES, dtype=torch.float64, device=device)
    total_union = torch.zeros(NUM_CLASSES, dtype=torch.float64, device=device)

    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Validate", leave=False):
            images = images.to(device, non_blocking=True)
            masks = masks.long().to(device, non_blocking=True)

            logits = forward_logits(model, images, target_shape=(masks.shape[-2], masks.shape[-1]))
            running_loss += float(criterion(logits, masks).item())

            preds = torch.argmax(logits, dim=1)
            intersection, union = compute_batch_intersection_union(
                pred=preds,
                target=masks,
                num_classes=NUM_CLASSES,
                ignore_index=IGNORE_INDEX,
            )
            total_intersection += intersection
            total_union += union

    avg_loss = running_loss / max(len(loader), 1)
    valid_classes = total_union > 0
    if torch.any(valid_classes):
        miou = float((total_intersection[valid_classes] / total_union[valid_classes]).mean().item())
    else:
        miou = 0.0

    return avg_loss, miou


def save_checkpoint(
    checkpoint_path: Path,
    model: Any,
    optimizer: Any,
    epoch: int,
    best_miou: float,
    global_step: int,
    config: Dict[str, Any],
) -> None:
    """Save model and optimizer state to disk."""
    import torch

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_miou": best_miou,
            "global_step": global_step,
            "config": config,
        },
        checkpoint_path,
    )


def maybe_resume(resume_path: Optional[Path], model: Any, optimizer: Any, device: Any) -> Tuple[int, float, int]:
    """Load checkpoint if provided and return (start_epoch, best_miou, step)."""
    import torch

    if resume_path is None:
        return 0, 0.0, 0
    if not resume_path.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

    checkpoint = torch.load(resume_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    start_epoch = int(checkpoint.get("epoch", -1)) + 1
    best_miou = float(checkpoint.get("best_miou", 0.0))
    global_step = int(checkpoint.get("global_step", 0))
    print(f"Resumed from {resume_path} | start_epoch={start_epoch} best_mIoU={best_miou:.4f}")
    return start_epoch, best_miou, global_step


def main() -> int:
    """Entrypoint for baseline training."""
    args = parse_args()
    yaml_config = load_yaml_config(args.config)
    run_cfg = resolve_config(args, yaml_config)

    import torch
    from torch import nn
    from torch.optim import Adam
    from torch.utils.tensorboard import SummaryWriter

    set_seed(run_cfg["seed"])

    data_dir: Path = run_cfg["data_dir"]
    output_dir: Path = run_cfg["output_dir"]
    checkpoints_dir = output_dir / "checkpoints"
    logs_dir = output_dir / "logs"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Using NUM_CLASSES from class_map: {NUM_CLASSES}")

    train_loader, val_loader = create_dataloaders(
        data_dir=data_dir,
        batch_size=run_cfg["batch_size"],
        num_workers=run_cfg["num_workers"],
        input_size=run_cfg["input_size"],
    )
    print(f"Train samples: {len(train_loader.dataset)} | Val samples: {len(val_loader.dataset)}")

    model = create_model(device=device)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    optimizer = Adam(model.parameters(), lr=run_cfg["lr"], weight_decay=run_cfg["weight_decay"])

    start_epoch, best_miou, global_step = maybe_resume(
        resume_path=run_cfg["resume"],
        model=model,
        optimizer=optimizer,
        device=device,
    )

    writer = SummaryWriter(log_dir=str(logs_dir))
    writer.add_text("run/config", json.dumps({k: str(v) for k, v in run_cfg.items()}, indent=2))

    start_time = time.time()
    try:
        for epoch in range(start_epoch, run_cfg["epochs"]):
            epoch_start = time.time()

            train_loss, global_step = train_one_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                writer=writer,
                epoch=epoch,
                global_step=global_step,
            )
            val_loss, val_miou = validate(model=model, loader=val_loader, criterion=criterion, device=device)

            writer.add_scalar("val/loss", val_loss, epoch)
            writer.add_scalar("val/mIoU", val_miou, epoch)

            print(
                f"Epoch [{epoch + 1}/{run_cfg['epochs']}] "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"val_mIoU={val_miou:.4f} time={time.time() - epoch_start:.1f}s"
            )

            save_checkpoint(
                checkpoint_path=checkpoints_dir / f"checkpoint_epoch_{epoch + 1}.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_miou=best_miou,
                global_step=global_step,
                config=run_cfg,
            )

            if val_miou > best_miou:
                best_miou = val_miou
                best_path = checkpoints_dir / "best.pt"
                save_checkpoint(
                    checkpoint_path=best_path,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    best_miou=best_miou,
                    global_step=global_step,
                    config=run_cfg,
                )
                print(f"Saved new best checkpoint: {best_path} (mIoU={best_miou:.4f})")
    finally:
        writer.close()

    print(
        f"Training complete. Best val mIoU: {best_miou:.4f}. "
        f"Total time: {(time.time() - start_time) / 60.0:.2f} min"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
