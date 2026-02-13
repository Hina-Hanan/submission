#!/usr/bin/env python3
"""
Production-ready object detection pipeline for gloved vs bare hand detection.
Uses YOLOv8 (Ultralytics) to detect and classify hands in images.
Outputs annotated images and JSON detection logs per image.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, List, Optional, Tuple

import cv2
from ultralytics import YOLO

# Output class names: map model class index to desired label
# NOTE: matches dataset/data.yaml -> names: ['bare_hand', 'gloved_hand']
CLASS_NAMES = {0: "bare_hand", 1: "gloved_hand"}


def iou(box1, box2) -> float:
    """Compute Intersection-over-Union for two [x1, y1, x2, y2] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0
    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    if area1 <= 0 or area2 <= 0:
        return 0.0
    return inter_area / float(area1 + area2 - inter_area + 1e-6)


def suppress_overlaps(detections: List[dict], iou_threshold: float = 0.7) -> List[dict]:
    """
    Suppress highly overlapping detections across classes.

    For regions where multiple boxes overlap strongly (IoU > threshold),
    keep only the highest-confidence detection. This reduces cases where
    the same hand is reported as both gloved and bare.
    """
    if len(detections) <= 1:
        return detections

    # Sort by confidence descending
    sorted_dets = sorted(detections, key=lambda d: d["confidence"], reverse=True)
    kept: List[dict] = []
    for det in sorted_dets:
        box = det["bbox"]
        if not kept:
            kept.append(det)
            continue
        overlap = False
        for k in kept:
            if iou(box, k["bbox"]) > iou_threshold:
                overlap = True
                break
        if not overlap:
            kept.append(det)
    return kept


def get_device() -> str:
    """
    Auto-detect best available device (GPU if CUDA available, else CPU).
    Returns device string for Ultralytics (e.g. '0', 'cpu').
    """
    try:
        import torch
        if torch.cuda.is_available():
            return "0"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"  # Apple Silicon
    except Exception:
        pass
    return "cpu"


def load_model(weights_path: str):
    """
    Load YOLOv8 model from weights file.

    Args:
        weights_path: Path to .pt weights file.

    Returns:
        Loaded YOLO model.

    Raises:
        FileNotFoundError: If weights file does not exist.
    """
    path = Path(weights_path)
    if not path.is_file():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    return YOLO(str(path))


def process_image(
    model,
    image_path: str,
    confidence: float,
    device: str,
) -> Tuple[Any, List[dict]]:
    """
    Run inference on a single image and return results plus detection records.

    Args:
        model: Loaded YOLO model.
        image_path: Path to input image.
        confidence: Minimum confidence threshold (0-1).
        device: Device string ('0', 'cpu', 'mps').

    Returns:
        (results object from model.predict, list of detection dicts for JSON).
    """
    results = model.predict(
        source=image_path,
        conf=confidence,
        device=device,
        verbose=False,
    )
    detections = []
    if results and len(results) > 0:
        r = results[0]
        if r.boxes is not None:
            h, w = r.orig_shape[:2]
            for box in r.boxes:
                cls_id = int(box.cls.item())
                conf = round(float(box.conf.item()), 2)
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, xyxy)
                # Clamp to image bounds
                x1 = max(0, min(x1, w))
                y1 = max(0, min(y1, h))
                x2 = max(0, min(x2, w))
                y2 = max(0, min(y2, h))
                label = CLASS_NAMES.get(cls_id, f"class_{cls_id}")
                detections.append(
                    {
                        "label": label,
                        "confidence": conf,
                        "bbox": [x1, y1, x2, y2],
                    }
                )
    detections = suppress_overlaps(detections)
    return results, detections


def annotate_image(results, image_path: str, output_path: str) -> None:
    """
    Draw bounding boxes and labels on image and save to output path.

    Args:
        results: Ultralytics results object (first element used).
        image_path: Original image path (used if results have no plot).
        output_path: Path to save annotated image.
    """
    if results and len(results) > 0:
        plotted = results[0].plot()
        cv2.imwrite(output_path, plotted)
    else:
        img = cv2.imread(image_path)
        if img is not None:
            cv2.imwrite(output_path, img)


def save_json(log_dir: str, filename: str, detections: List[dict]) -> None:
    """
    Write detection log for one image to a JSON file in log_dir.

    Args:
        log_dir: Directory for log files (created if missing).
        filename: Image filename (e.g. 'image1.jpg').
        detections: List of detection dicts (label, confidence, bbox).
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_name = Path(filename).stem + ".json"
    log_path = Path(log_dir) / log_name
    payload = {"filename": filename, "detections": detections}
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def process_single(args: argparse.Namespace, image_path: Path, model, device: str) -> None:
    """Process one image: inference, annotate, save JSON."""
    results, detections = process_image(
        model, str(image_path), args.confidence, device
    )
    out_img = Path(args.output) / image_path.name
    annotate_image(results, str(image_path), str(out_img))
    save_json(args.logs, image_path.name, detections)


def run_batch_inference(
    model,
    image_paths: List[Path],
    confidence: float,
    device: str,
) -> List[tuple]:
    """
    Run inference on a batch of images. Returns list of (results, detections_list) per image.
    """
    paths_str = [str(p) for p in image_paths]
    results_list = model.predict(
        source=paths_str,
        conf=confidence,
        device=device,
        verbose=False,
    )
    out = []
    for i, (res, path) in enumerate(zip(results_list or [], image_paths)):
        detections = []
        if res and res.boxes is not None:
            h, w = res.orig_shape[:2]
            for box in res.boxes:
                cls_id = int(box.cls.item())
                conf = round(float(box.conf.item()), 2)
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, xyxy)
                x1, y1 = max(0, min(x1, w)), max(0, min(y1, h))
                x2, y2 = max(0, min(x2, w)), max(0, min(y2, h))
                label = CLASS_NAMES.get(cls_id, f"class_{cls_id}")
                detections.append(
                    {"label": label, "confidence": conf, "bbox": [x1, y1, x2, y2]}
                )
        detections = suppress_overlaps(detections)
        out.append((res, detections, path))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run hand (gloved/bare) detection on a folder of images."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input folder containing .jpg images",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="Output folder for annotated images (default: output)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Confidence threshold 0-1 (default: 0.5)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to YOLOv8 model weights (.pt)",
    )
    parser.add_argument(
        "--logs",
        type=str,
        default="logs",
        help="Folder for JSON detection logs (default: logs)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="Batch size for inference (default: 8). Use 1 for sequential.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    if not input_dir.is_dir():
        print(f"Error: Input path is not a directory: {args.input}", file=sys.stderr)
        sys.exit(1)

    image_ext = {".jpg", ".jpeg", ".JPG", ".JPEG"}
    image_paths = [p for p in input_dir.iterdir() if p.is_file() and p.suffix in image_ext]
    if not image_paths:
        print(f"Error: No .jpg images found in {args.input}", file=sys.stderr)
        sys.exit(1)

    try:
        model = load_model(args.weights)
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

    device = get_device()
    print(f"Using device: {device}")

    Path(args.output).mkdir(parents=True, exist_ok=True)
    Path(args.logs).mkdir(parents=True, exist_ok=True)

    batch_size = max(1, args.batch)
    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        if batch_size == 1:
            for path in batch_paths:
                process_single(args, path, model, device)
        else:
            batch_results = run_batch_inference(
                model, batch_paths, args.confidence, device
            )
            for res, detections, path in batch_results:
                out_img = Path(args.output) / path.name
                annotate_image([res], str(path), str(out_img))
                save_json(args.logs, path.name, detections)

    print(f"Processed {len(image_paths)} image(s). Annotated images -> {args.output}, logs -> {args.logs}")


if __name__ == "__main__":
    main()
