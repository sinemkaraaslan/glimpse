# Glimpse 🌸

A face search tool that finds a person in video footage. Upload a reference photo and a video — Glimpse scans through and shows you every scene where that face appears.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red)

## Features

- Face detection and matching using deep learning
- Scene grouping (consecutive matches shown together)
- Confidence scores for each detection
- Adjustable tolerance threshold
- Sensitivity curve visualization

## Installation

```bash
# Clone the repo
git clone https://github.com/sinemkaraaslan/glimpse.git
cd glimpse

# Create environment
conda create -n glimpse python=3.10
conda activate glimpse

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
streamlit run app.py
```

Then open `localhost:8501` in your browser.

1. Upload a reference face (jpg/png)
2. Upload a video to analyze (mp4/mov/avi)
3. Adjust tolerance if needed
4. Wait for results

## How it works

Glimpse uses the `face_recognition` library (built on dlib) to encode faces into 128-dimensional vectors. It compares the reference face against every face found in the video and groups consecutive matches into scenes.
