# Hand Detection Pipeline (Gloved vs Bare)

Production-ready object detection pipeline to detect hands in images and classify them as **gloved_hand** or **bare_hand**. Built with YOLOv8 (Ultralytics), PyTorch, and OpenCV.

## Table of Contents

- [Installation](#installation)
- [How to Run](#how-to-run) ⭐ **Start Here**
- [Dataset](#dataset)
- [Model](#model)
- [Training](#training)
- [Inference](#inference)
- [Project structure](#project-structure)
- [What worked](#what-worked)
- [Limitations and what didn't work as well](#limitations-and-what-didnt-work-as-well)
- [Future improvements](#future-improvements)

---

## Installation

### Mac (macOS) Users

**Good news:** All commands are the same! The pipeline automatically detects and uses Apple Silicon GPU (MPS) if available.

#### Additional Setup for Mac:

1. **Python Installation:**
   - If you don't have Python 3.8+, install via Homebrew:
     ```bash
     brew install python3
     ```
   - Or download from [python.org](https://www.python.org/downloads/)

2. **Create Virtual Environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Note: use 'source' instead of Windows' 'venv\Scripts\activate'
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   
   **Note:** PyTorch will automatically use MPS (Metal Performance Shaders) on Apple Silicon Macs (M1/M2/M3) for faster inference and training. The script auto-detects this - no additional configuration needed!

4. **Commands are identical:**
   - Training: Same commands as Windows/Linux
   - Inference: Same commands as Windows/Linux
   - The only difference is using `python3` instead of `python` if your system requires it

#### Apple Silicon (M1/M2/M3) Performance:

- **Training:** MPS acceleration provides faster training than CPU-only, but slower than NVIDIA GPU
- **Inference:** Significantly faster than CPU-only
- **Recommended settings for Apple Silicon:**
  ```bash
  python train.py --data dataset/data.yaml --model yolov8n.pt --epochs 25 --batch 8 --imgsz 640 --weights-dir weights
  ```
  (Batch size 8-12 works well on Apple Silicon)

### Windows/Linux Users

Standard installation:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

---

## How to Run

### Quick Start Guide for Reviewers

This section shows you exactly how to run the scripts. Choose one option below.

---

### Option 1: Train and Test (Recommended)

**Use this if:** You want to verify the training pipeline works correctly.

#### Step 1: Train the Model

```bash
# Navigate to project folder
cd Part_1_Glove_Detection

# Activate virtual environment (if using one)
# Windows: venv\Scripts\activate
# Mac/Linux: source venv/bin/activate

# Run training
python train.py --data dataset/data.yaml --model yolov8n.pt --epochs 12 --batch 2 --imgsz 320 --weights-dir weights
```

**What happens:**
- Training starts (takes 30 minutes to several hours)
- Creates/overwrites `train_run/` folder with new weights
- Saves model to `train_run/weights/best.pt`

#### Step 2: Test with Your Images

```bash
# Create a folder with your test images
mkdir my_test_images
# (Add your .jpg images to my_test_images folder)

# Run detection
python detection_script.py --input my_test_images --output results --weights train_run/weights/best.pt --confidence 0.5 --logs logs --batch 4
```

**Results:**
- Annotated images → `results/` folder
- Detection logs → `logs/` folder (JSON files)

---

### Option 2: Quick Test Only (Use Pre-trained Weights)

**Use this if:** You just want to test inference quickly without training.

#### Step 1: Verify Weights Exist

```bash
cd Part_1_Glove_Detection
ls train_run/weights/best.pt  # Should show the file
```

#### Step 2: Run Detection

```bash
# Create folder with your test images
mkdir my_test_images
# (Add your .jpg images to my_test_images folder)

# Run detection
python detection_script.py --input my_test_images --output results --weights train_run/weights/best.pt --confidence 0.5 --logs logs --batch 4
```

**Results:**
- Annotated images → `results/` folder
- Detection logs → `logs/` folder (JSON files)

---

### Command Parameters Explained

**For `detection_script.py`:**
- `--input my_test_images` → Folder containing your `.jpg` images
- `--output results` → Where annotated images will be saved
- `--weights train_run/weights/best.pt` → Model weights file (required)
- `--confidence 0.5` → Detection threshold (0-1, default: 0.5)
- `--logs logs` → Where JSON logs will be saved
- `--batch 4` → Batch size (use 1 for very limited memory)

**For `train.py`:**
- `--data dataset/data.yaml` → Dataset configuration file
- `--model yolov8n.pt` → Base model to use
- `--epochs 12` → Number of training epochs
- `--batch 2` → Batch size (increase if you have GPU)
- `--imgsz 320` → Image size (use 640 for better results if you have GPU)
- `--weights-dir weights` → Where to save weights

---

### Troubleshooting

**Error: "Weights file not found"**
- Solution: Train the model first (Option 1) or check that `train_run/weights/best.pt` exists

**Error: "No images found"**
- Solution: Make sure your images are `.jpg` or `.jpeg` format and in the folder you specified

**Error: "Module not found"**
- Solution: Install dependencies: `pip install -r requirements.txt`

**Training is slow**
- Solution: Use smaller batch size (`--batch 1` or `--batch 2`) and lower image size (`--imgsz 320`)

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

**Mac users (Apple Silicon):** MPS acceleration provides a middle ground - faster than CPU-only but slower than NVIDIA GPU. You can use batch sizes of 8-12 and higher resolutions (640) for better results than CPU-only training.

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

3. **Training outputs:** 
   - A `train_run/` folder is created automatically (you don't need it beforehand)
   - **Important:** If `train_run/` already exists (e.g., from cloning the repo), it will be **overwritten** with new training results
   - Best weights are saved to `train_run/weights/best.pt`
   - Weights are also copied to `weights/best.pt` (if `--weights-dir weights` is used)
   - Training plots, logs, and configuration files are saved in `train_run/`
   
   **For inference:** Use `train_run/weights/best.pt` or `weights/best.pt` - both contain the same model.
   
   **Note for reviewers:** If you clone this repository and `train_run/` already exists, running training will replace the existing folder. You will NOT get duplicate folders - only one `train_run/` will exist at a time.

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

### GPU/CPU/MPS (Apple Silicon)

The script auto-detects device in this priority order:
1. **CUDA GPU** (NVIDIA) - if available
2. **MPS** (Apple Silicon Mac M1/M2/M3) - if available
3. **CPU** - fallback

**Mac users:** MPS acceleration is automatically used on Apple Silicon Macs - no additional installation or configuration needed! Commands are identical to Windows/Linux.

---

## Testing with Your Own Images

**📌 All instructions for running the scripts are in the [How to Run](#how-to-run) section above.**

### About Pre-trained Weights

When you clone this repository, you'll see a `train_run/` folder containing my pre-trained model weights:
- `train_run/weights/best.pt` - The trained model file
- Training plots and logs from my training session

**Note:** If you run training yourself, it will overwrite this folder (this is safe - you can always re-clone to get original weights back).

---

## Project structure

```
Part_1_Glove_Detection/
├── dataset/
│   ├── train/  (images + labels)
│   ├── valid/
│   ├── test/
│   └── data.yaml
├── train_run/         # Created automatically during training (optional - only needed if using pre-trained weights)
│   └── weights/
│       └── best.pt    # Trained model weights (needed for inference)
├── output/            # Annotated images (created during inference)
├── logs/              # JSON detection logs (created during inference)
├── detection_script.py
├── train.py 
├── requirements.txt
└── README.md
```

**Note:** 
- `train_run/` is created automatically when you run training - you don't need it beforehand
- For inference, you need `train_run/weights/best.pt` (or train fresh to create it)
- `output/` and `logs/` are created automatically during inference

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

