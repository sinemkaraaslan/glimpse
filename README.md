# Glimpse

**Face search and scene-based video analysis system.**

Glimpse locates a specific person within video footage using a single reference image. Unlike frame-level face detection demos, the system performs scene-based temporal aggregation and confidence-aware reporting, making it suitable for real-world video analysis scenarios.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)

## Motivation

Manual inspection of long video recordings is time-consuming and error-prone. Glimpse automates this by combining classical computer vision with deep learning-based facial embeddings:

- One-shot identity search (single reference photo)
- Scene-level reporting with start–end timestamps
- Adjustable sensitivity and performance analysis

## How It Works

The system follows a hybrid pipeline optimized for CPU-based processing. The pipeline prioritizes interpretability, efficiency, and temporal consistency over raw frame-level detection.

**1. Face Detection** — HOG (Histogram of Oriented Gradients), chosen for low computational cost and CPU robustness.

**2. Face Alignment & Feature Extraction** — Pre-trained ResNet-34 model (via `dlib` / `face_recognition`) produces a 128-dimensional embedding per detected face.

**3. Biometric Matching** — Euclidean (L2) distance between embeddings. Only a single reference image is needed (one-shot learning).

**4. Confidence Scoring** — Non-linear distance-to-confidence transformation for user-readable similarity percentages.

**5. Scene-Based Aggregation** — Consecutive matching frames are grouped into scenes, each reported with start–end timestamps and best confidence score.

## Features

- One-shot face search — no model training required
- Scene-level temporal reporting (not just individual frames)
- Scene-level confidence tracking instead of frame-wise decisions
- Adjustable similarity threshold (tolerance)
- Sensitivity analysis visualization
- Streamlit-based interactive UI
- Runs on CPU only, no GPU required

## Performance

| Resolution | Mode | Avg FPS |
|---|---|---|
| 1920×1080 | Original | ~5.8 |
| 640×360 | Resized | ~14.4 |

Downscaling improves performance by roughly 2.5× with minimal accuracy loss. Empirically, a threshold around 0.6 provides the best trade-off between false positives and missed detections.

## Tech Stack

- Python 3.10
- OpenCV
- dlib / face_recognition
- Streamlit
- NumPy, Pandas
- Altair
- Pillow

## Usage

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Limitations

- HOG-based detection may struggle with very small or heavily occluded faces
- Performance depends on reference image quality
- Not designed for large-scale identity databases

## Future Work

- CNN-based face detection (MTCNN or RetinaFace)
- Precision–Recall and ROC curve evaluation
- Exportable structured reports (CSV / JSON)
- Multi-identity search support

## Academic Context

Originally developed as part of a Computer Vision course, with an accompanying technical report covering face recognition literature, model selection rationale, and experimental evaluation. The GitHub version focuses on clarity, reproducibility, and practical usability.
