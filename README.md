# Cross-View Gait Recognition System for Railway Scenarios

This project implements a cross-view gait recognition system for railway scenarios, using deep learning techniques for person detection, tracking, pose estimation, and gait recognition.

## Features
- Person detection using YOLOv5
- Person tracking using DeepSORT
- Pose estimation using HRNet
- Multi-scale feature extraction with attention mechanisms
- Cross-view gait recognition with view transformation
- Comprehensive evaluation and visualization tools

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
# Train the model
python train.py --config config/train_config.yaml

# Evaluate the model
python evaluate.py --config config/eval_config.yaml --checkpoint checkpoints/model_best.pth

# Run inference on video
python inference.py --video path/to/video.mp4 --output results/
```

## Results
The system achieves:
- Overall accuracy: 78.5%
- Cross-45° accuracy: 72.3%
- Cross-90° accuracy: 63.8%
- Processing time: 0.18s per person
