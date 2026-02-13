# Hand Detection Pipeline (Gloved vs Bare)

Production-ready object detection pipeline to detect hands in images and classify them as **gloved_hand** or **bare_hand**. Built with YOLOv8 (Ultralytics), PyTorch, and OpenCV.

---

## Dataset

- **Name:** Glove Hand and Bare Hand (Roboflow)
- **Source:** [Roboflow Universe – Glove Hand and Bare Hand](https://universe.roboflow.com/glove-detection-3vldq/glove-hand-and-bare-hand-zwvif/dataset/3)
- **License:** CC BY 4.0
- **Classes:** 2  
- `0`: bare_hand  
- `1`: gloved_hand
- **Format:** YOLO (normalized `class x_center y_center width height` per object)

### Layout

```
dataset/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── data.yaml
```

Paths in `data.yaml` are relative to the `dataset/` directory (`train/images`, `valid/images`, `test/images`).

---

## Model

- **Architecture:** YOLOv8 (Ultralytics)
- **Default base:** `yolov8n.pt` (nano); can use `yolov8s.pt`, `yolov8m.pt`, etc.
- **Output classes:** `bare_hand`, `gloved_hand` (matches `data.yaml` class order)

---

## Training

### Config

- **Epochs:** 25 (default; 20–30 recommended)
- **Image size:** 640
- **Batch size:** 16 (tune by GPU memory)
- **Augmentations:** enabled by default in Ultralytics, with explicit tuning:
  - **Horizontal flip:** 0.5
  - **HSV:** hue 0.015, saturation 0.7, value 0.4
  - **Scaling:** 0.5
  - **Translation:** 0.1
  - **Mosaic:** 1.0

### How to train

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. From the project root (e.g. `Part_1_Glove_Detection/`), run:

   ```bash
   # GPU or strong CPU (default config)
   python train.py --data dataset/data.yaml --model yolov8n.pt --epochs 25 --batch 16 --weights-dir weights
   ```

   On a **CPU-only laptop**, a lighter but still effective config is:

   ```bash
   python train.py --data dataset/data.yaml --model yolov8n.pt --epochs 5 --batch 2 --imgsz 320 --weights-dir weights
   ```

   Optional: use a larger model and custom paths:

   ```bash
   python train.py --data dataset/data.yaml --model yolov8s.pt --epochs 30 --imgsz 640 --batch 8 --weights-dir weights --project .
   ```

3. Best weights are written to `weights/best.pt` and also stored under the latest run directory (e.g. `train_run/weights/best.pt`). In practice you can use either path; this README uses the `train_run/weights/best.pt` location for inference.

### Training command snippet

```bash
cd Part_1_Glove_Detection
python train.py --data dataset/data.yaml --model yolov8n.pt --epochs 25 --batch 16 --weights-dir weights
```

---

## Inference

### How to run

From the project root:

```bash
python detection_script.py --input /path/to/images --output output --weights train_run/weights/best.pt [--confidence 0.5] [--logs logs] [--batch 8]
```

- **`--input`** (required): folder containing `.jpg` images  
- **`--output`**: folder for annotated images (default: `output`)  
- **`--weights`** (required): path to `.pt` model file  
- **`--confidence`**: detection threshold 0–1 (default: 0.5)  
- **`--logs`**: folder for per-image JSON logs (default: `logs`)  
- **`--batch`**: batch size for inference (default: 8; use 1 for sequential)

### Example CLI usage

```bash
# Default confidence 0.5, batch 8
python detection_script.py --input ./dataset/test/images --output ./output --weights ./train_run/weights/best.pt

# Higher confidence, custom log dir
python detection_script.py --input ./my_photos --output ./annotated --weights ./train_run/weights/best.pt --confidence 0.6 --logs ./detection_logs

# Sequential (batch=1)
python detection_script.py --input ./images --output ./out --weights ./train_run/weights/best.pt --batch 1
```

### Outputs

1. **Annotated images** in `--output`: input images with bounding boxes and labels.  
2. **JSON logs** in `--logs`: one file per image, same base name with `.json`.

**JSON format (per image):**

```json
{
  "filename": "image1.jpg",
  "detections": [
    {
      "label": "gloved_hand",
      "confidence": 0.92,
      "bbox": [x1, y1, x2, y2]
    }
  ]
}
```

- **bbox:** absolute pixel coordinates `[x1, y1, x2, y2]`.  
- **confidence:** rounded to 2 decimal places.

### GPU/CPU

The script auto-detects device: CUDA GPU if available, else MPS (Apple Silicon), else CPU. No extra flags required.

---

## Project structure

```
Part_1_Glove_Detection/
├── dataset/
│   ├── train/  (images + labels)
│   ├── valid/
│   ├── test/
│   └── data.yaml
├── weights/           # best.pt, etc.
├── output/            # annotated images
├── logs/              # JSON detection logs
├── detection_script.py
├── train.py
├── requirements.txt
└── README.md
```

---

## What worked

- YOLOv8n training on the Roboflow Gloves dataset with 25 epochs and default augmentations gives a good speed/accuracy tradeoff.
- Batch inference (`--batch 8`) speeds up processing on GPU.
- Explicit HSV and flip augmentations improve robustness to lighting and orientation.
- Auto device selection (GPU/CPU/MPS) keeps the same command across machines.

---

## Limitations

- Performance depends on dataset size and diversity; small or biased data may not generalize to all environments.
- Very small or heavily occluded hands may be missed or misclassified.
- Two-class only (gloved vs bare); no finer glove types.
- JSON logs are per image only; no global summary JSON (can be added as a post-step).

---

## Future improvements

- Add a summary JSON (e.g. `detections_summary.json`) with counts and aggregate stats.
- Optional multiprocessing or threading for I/O-bound steps (batch inference already improves throughput).
- Export to ONNX/TensorRT for deployment.
- Support video input and webcam.
- Confidence calibration and precision/recall reporting on a fixed test set.

---

