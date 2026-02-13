# Hand Detection Pipeline (Gloved vs Bare)

Production-ready object detection pipeline to detect hands in images and classify them as **gloved_hand** or **bare_hand**. Built with YOLOv8 (Ultralytics), PyTorch, and OpenCV.

## Table of Contents

- [Dataset](#dataset)
- [Model](#model)
- [Training](#training)
- [Inference](#inference)
- [Project structure](#project-structure)
- [What worked](#what-worked)
- [Limitations and what didn't work as well](#limitations-and-what-didnt-work-as-well)
- [Future improvements](#future-improvements)

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

### Hardware Constraints

**Note:** This model was trained on a **CPU-only laptop** due to hardware limitations. Training would have been significantly faster and potentially more accurate with GPU acceleration (e.g., Google Colab GPU or a local CUDA GPU). With GPU, I could have:
- Trained for more epochs (20-30 instead of 12) in reasonable time
- Used larger batch sizes (16-32 instead of 2) for better gradient estimates
- Used higher image resolution (640 instead of 320) for finer detail
- Trained larger models (yolov8s or yolov8m) without excessive wait times

Despite CPU constraints, the model achieves reasonable performance through careful hyperparameter tuning and lightweight configurations.

### Training Configuration Used

- **Epochs:** 12 (limited by CPU training time)
- **Image size:** 320 (reduced from 640 to fit CPU memory)
- **Batch size:** 2 (minimal to avoid out-of-memory errors)
- **Model:** YOLOv8n (nano - smallest variant)
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

2. From the project root (e.g. `Part_1_Glove_Detection/`), choose the appropriate command:

   **⭐ RECOMMENDED: GPU or Strong CPU (for best results)**
   
   This configuration will produce higher confidence scores and better generalization:
   ```bash
   python train.py --data dataset/data.yaml --model yolov8n.pt --epochs 25 --batch 16 --imgsz 640 --weights-dir weights
   ```
   
   **What I actually used (CPU-only laptop):**
   
   Due to hardware constraints, I used a lightweight configuration:
   ```bash
   python train.py --data dataset/data.yaml --model yolov8n.pt --epochs 12 --batch 2 --imgsz 320 --weights-dir weights
   ```
   
   **Optional: Larger model for even better accuracy (requires GPU):**
   ```bash
   python train.py --data dataset/data.yaml --model yolov8s.pt --epochs 30 --imgsz 640 --batch 8 --weights-dir weights --project .
   ```

3. Best weights are written to `weights/best.pt` and also stored under the latest run directory (e.g. `train_run/weights/best.pt`). In practice you can use either path; this README uses the `train_run/weights/best.pt` location for inference.

### Training commands summary

**Preferred (GPU/Strong CPU):**
```bash
cd Part_1_Glove_Detection
python train.py --data dataset/data.yaml --model yolov8n.pt --epochs 25 --batch 16 --imgsz 640 --weights-dir weights
```

**What was used in this project (CPU-only):**
```bash
cd Part_1_Glove_Detection
python train.py --data dataset/data.yaml --model yolov8n.pt --epochs 12 --batch 2 --imgsz 320 --weights-dir weights
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

**⭐ Recommended (GPU or Strong CPU):**
```bash
python detection_script.py --input dataset/test/images --output output --weights train_run/weights/best.pt --confidence 0.5 --logs logs --batch 8
```

**What I used (CPU-only laptop):**
```bash
python detection_script.py --input dataset/test/images --output output --weights train_run/weights/best.pt --confidence 0.5 --logs logs --batch 4
```

**Other examples:**
```bash
# Higher confidence threshold
python detection_script.py --input ./my_photos --output ./annotated --weights ./train_run/weights/best.pt --confidence 0.6 --logs ./detection_logs --batch 8

# Sequential processing (batch=1) - useful for very memory-constrained systems
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

### Positive outcomes

- **YOLOv8n fine-tuning:** Successfully fine-tuned a pretrained COCO model on the Roboflow Glove Hand and Bare Hand dataset, adapting it to the two-class hand detection task.
- **CPU training feasibility:** Despite hardware constraints, achieved functional model training using lightweight settings (epochs 12, batch 2, imgsz 320). The model successfully detects and classifies hands in test images.
- **Augmentation strategy:** Explicit HSV adjustments (hue, saturation, value) and horizontal flip augmentations improved robustness to different lighting conditions and hand orientations.
- **Batch inference:** Implemented batch processing (`--batch 4`) for faster inference compared to sequential processing, even on CPU.
- **IoU-based post-processing:** Added overlap suppression to remove duplicate detections when the same hand was detected as both gloved and bare, improving output quality.
- **Auto device detection:** The pipeline automatically selects the best available device (GPU → MPS → CPU), making it portable across different machines.
- **Clean CLI interface:** Well-structured command-line arguments (`--input`, `--output`, `--confidence`, `--weights`, `--logs`, `--batch`) make the script easy to use and integrate.
- **Comprehensive output:** Both annotated images and structured JSON logs per image provide multiple ways to analyze detection results.

### Improvements made during development

- **Class mapping fix:** Corrected class index mapping to match dataset labels (`0: bare_hand`, `1: gloved_hand`).
- **Overlap suppression:** Implemented IoU-based filtering to handle cases where overlapping boxes with different class labels appeared on the same hand.
- **CPU-optimized training:** Adjusted hyperparameters (smaller batch, lower resolution, fewer epochs) to work within CPU memory and time constraints while maintaining reasonable accuracy.

---

## Limitations and what didn't work as well

### Hardware-related limitations

- **CPU training constraints:** Training on CPU limited the number of epochs (12 vs ideal 20-30), batch size (2 vs ideal 16+), and image resolution (320 vs ideal 640). This likely reduced model confidence and generalization compared to GPU training.
- **Lower confidence scores:** Due to limited training, the model sometimes produces lower confidence scores (0.3-0.5 range) on images outside the training distribution, especially web stock photos with different lighting/backgrounds.
- **Training time:** CPU training took significantly longer than GPU would have, limiting experimentation with different hyperparameters.

### Model and data limitations

- **Domain shift:** The model performs well on images similar to the training set but struggles with significantly different styles (e.g., professional stock photos, unusual camera angles, different backgrounds).
- **Class imbalance:** The dataset had more gloved hand examples than bare hands, which may have affected the model's ability to confidently detect bare hands in some scenarios.
- **Small dataset:** With limited training data, the model may not generalize well to all real-world factory environments without additional domain-specific training.
- **Occlusion and scale:** Very small or heavily occluded hands may be missed or misclassified.
- **Two-class limitation:** Only detects gloved vs bare; cannot distinguish between different glove types or other PPE items.

### Technical limitations

- **No global summary:** JSON logs are per-image only; no aggregate statistics or summary JSON file (can be added as a post-processing step).
- **Confidence threshold sensitivity:** The model's performance is sensitive to the confidence threshold setting; lower thresholds catch more hands but may include false positives.

---

## Future improvements

### Immediate improvements (with GPU access)

- **Extended training:** Train for 20-30 epochs with larger batch sizes (16-32) and higher resolution (640) to improve model confidence and accuracy.
- **Larger model:** Experiment with YOLOv8s or YOLOv8m for better detection performance, especially on challenging images.
- **More training data:** Add diverse examples (different lighting, backgrounds, camera angles) to improve generalization, especially for bare hand detection.
- **Class balancing:** Ensure equal representation of gloved and bare hands in the training set to improve detection confidence for both classes.

### Technical enhancements

- **Summary statistics:** Add a global summary JSON (e.g. `detections_summary.json`) with aggregate counts, average confidence scores, and class distribution across all processed images.
- **Confidence calibration:** Implement confidence score calibration to better align predicted probabilities with actual accuracy.
- **Performance metrics:** Add precision/recall reporting on a fixed test set to quantitatively measure model performance.
- **Video support:** Extend the pipeline to process video streams or webcam input for real-time monitoring applications.
- **Model optimization:** Export to ONNX or TensorRT for faster inference in production deployments.
- **Advanced post-processing:** Implement temporal smoothing for video inputs to reduce flickering detections across frames.

### Deployment considerations

- **API wrapper:** Create a REST API or web service wrapper for easy integration into factory monitoring systems.
- **Alerting system:** Add functionality to trigger alerts when bare hands are detected in safety-critical zones.
- **Database integration:** Store detection logs in a database for historical analysis and compliance reporting.

---

