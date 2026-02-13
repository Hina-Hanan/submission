#!/usr/bin/env python3
"""
YOLOv8 training script for gloved vs bare hand detection.
Uses Ultralytics YOLOv8 with augmentation. Saves best weights to weights/.
"""

import argparse
import sys
from pathlib import Path

from ultralytics import YOLO


# Augmentation and training defaults (YOLOv8 built-in + explicit for docs)
# Ultralytics applies: hsv_h, hsv_s, hsv_v, flipud, fliplr, mosaic, scale, etc.
DEFAULT_EPOCHS = 25
DEFAULT_IMGSZ = 640
DEFAULT_BATCH = 16
DEFAULT_MODEL = "yolov8n.pt"


def get_data_yaml(project_root: Path) -> Path:
    """Resolve dataset/data.yaml from project root or cwd."""
    for base in (project_root, Path.cwd()):
        yaml = base / "dataset" / "data.yaml"
        if yaml.exists():
            return yaml
    return project_root / "dataset" / "data.yaml"


def main() -> int:
    parser = argparse.ArgumentParser(description="Train YOLOv8 for hand (gloved/bare) detection.")
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to data.yaml (default: dataset/data.yaml)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Base model (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Number of epochs (default: {DEFAULT_EPOCHS})",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=DEFAULT_IMGSZ,
        help=f"Image size (default: {DEFAULT_IMGSZ})",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=DEFAULT_BATCH,
        help=f"Batch size (default: {DEFAULT_BATCH})",
    )
    parser.add_argument(
        "--weights-dir",
        type=str,
        default="weights",
        help="Directory to save best.pt and last.pt",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=".",
        help="Project root (default: current dir)",
    )
    args = parser.parse_args()

    project_root = Path(args.project).resolve()
    data_yaml = args.data
    if not data_yaml:
        data_yaml = str(get_data_yaml(project_root))
    if not Path(data_yaml).exists():
        print(f"Error: data.yaml not found: {data_yaml}", file=sys.stderr)
        return 1

    weights_dir = Path(args.weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)

    # Train with augmentations enabled (default in Ultralytics):
    # - horizontal flip (hflip)
    # - vertical flip (vflip)
    # - mosaic, mixup
    # - HSV (hue, saturation, value)
    # - scaling, translation
    results = model.train(
        data=data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(project_root),
        name="train_run",
        exist_ok=True,
        augment=True,
        # Explicit augmentation-related hyperparameters (optional override)
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
    )

    # Copy best weights to weights/
    save_dir = Path(results.save_dir)
    best_pt = save_dir / "best.pt"
    if best_pt.exists():
        import shutil
        dest = weights_dir / "best.pt"
        shutil.copy(best_pt, dest)
        print(f"Best weights saved to {dest}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
