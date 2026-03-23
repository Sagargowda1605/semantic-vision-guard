# Semantic Vision Guard 

A real-time video perception pipeline that combines **YOLOv8** for object 
detection and **CLIP** for semantic verification — acting as a "guard" 
against false detections.

## How it works
```
Video Frame → YOLOv8 Detection → ROI Extraction → CLIP Verification → Annotated Output
```

YOLO detects objects and extracts regions of interest (ROIs).
CLIP independently verifies each ROI using zero-shot semantic scoring.
Detections with low CLIP confidence are flagged as suspicious.

## Why this matters

YOLO alone can produce false detections — especially in challenging conditions
like snow, occlusion, or low resolution. CLIP acts as a second opinion,
reducing false positives using vision-language understanding.

## Tech Stack

- YOLOv8 (Ultralytics) — object detection
- CLIP ViT-B/32 (OpenCLIP) — zero-shot semantic verification  
- OpenCV — video processing and visualization
- FastAPI — REST API for inference (coming soon)
- ONNX/TensorRT — optimized deployment (coming soon)

## Project Structure
```
semantic-vision-guard/
├── app/
│   └── models/
│       ├── yolo.py       # YOLOv8 detection + COCO classes
│       ├── clip.py       # CLIP semantic scorer
│       └── opencv.py     # Video pipeline + visualization
├── data/
│   └── input/            # Input videos
├── requirements.txt
└── README.md
```

## Setup
```bash
git clone https://github.com/Sagargowda1605/semantic-vision-guard
cd semantic-vision-guard
python -m venv .yolo
.yolo\Scripts\activate  # Windows
pip install -r requirements.txt
```

## Run
```bash
python app/models/opencv.py
```

## Results

- Green box ✓ — YOLO + CLIP both confident → verified detection
- Red box ✗ — CLIP score below threshold → suspicious detection

CLIP scores for real objects are consistent (0.22–0.27) while false 
detections show lower and inconsistent scores — validating the guard concept.

## Roadmap

- [x] YOLOv8 ROI extraction
- [x] CLIP semantic verification  
- [x] Real-time video pipeline
- [ ] FastAPI REST endpoint
- [ ] ONNX export for optimized inference
- [ ] TensorRT deployment

## Author

Sagar Mattikere Anand  
M.Sc. Computer Science, Philipps University Marburg  
[LinkedIn](https://www.linkedin.com/in/sagar-mattikere-anand-273947245/) · 
[GitHub](https://github.com/Sagargowda1605)