# Retail Security Object Detection with YOLOv8s

---
## All the required complete deliverables are available in the drive link - https://drive.google.com/drive/folders/1yHOP7Zhe-UK7W8C3qLkw0AiKcCc0ohTx?usp=sharing

## Environment Setup

### Dependencies
All required packages are listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```

**Key packages:**
- ultralytics
- opencv-python
- torch
- numpy
- matplotlib
- pyyaml

### Compute
Trained on Google Colab with T4 GPU.

---

## Labeling Process

### Step 1: Extract Frames
Extracted frames from the two videos provided:

### Step 2: Initial Labeling with Grounding DINO (Didn't include the script in the final notebook for redundancy) 
Used Grounding DINO for automated initial detection:
```bash
pip install groundingdino-py
```
```python
from groundingdino.util.inference import load_model, load_image, predict

model = load_model("config.py", "weights.pth")
TEXT_PROMPT = "person . backpack . bottle . refrigerator"

boxes, logits, phrases = predict(
    model=model,
    image=image,
    caption=TEXT_PROMPT,
    box_threshold=0.35,
    text_threshold=0.25
)
```

### Step 3: Manual Refinement in Roboflow
- Uploaded frames with DINO labels to Roboflow
- Manually verified and corrected bounding boxes
- Exported dataset in YOLOv8 format

### Step 4: Data Augmentation
Addressed class imbalance by augmenting bottle images:
```python
def augment_image(img_path):
    img = cv2.imread(img_path)
    # Applied flip, brightness, contrast, HSV, blur, noise
    return augmented_img
```

## Training

### Download Dataset
Download datasets from Google Drive:
https://drive.google.com/drive/folders/1yHOP7Zhe-UK7W8C3qLkw0AiKcCc0ohTx?usp=sharing

Extract to the project directory.

### Train Model

Using the notebook provided:
Open `Final_notebook.ipynb` and run all cells(you can skip the ones with extracting frames and training since we already have the weights).

### Training Parameters
- Model: yolov8s.pt
- Epochs: 100 (early stopping patience=15)
- Image size: 640x640
- Batch size: 16


## Inference

### Download Model Weights
From the google drive link below, download the best.pt and place it in the folder and run the inference seciton in the notebook.
[https://drive.google.com/drive/folders/1yHOP7Zhe-UK7W8C3qLkw0AiKcCc0ohTx?usp=sharing](https://drive.google.com/file/d/1aDsnjXSU8oVBngkSk9RVQwwjm7zdgPMH/view?usp=sharing)


