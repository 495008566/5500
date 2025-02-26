# Cross-View Gait Recognition System for Railway Scenarios

This repository contains the implementation of a cross-view gait recognition system designed specifically for railway scenarios. The system uses deep learning techniques to identify individuals across different camera viewing angles.

## Features

- Person detection using YOLOv5
- Person tracking using DeepSORT
- Pose estimation using HRNet
- Multi-scale feature extraction with attention mechanisms
- Cross-view gait recognition with view transformation
- Comprehensive evaluation and visualization tools

## Project Structure

```
gait_recognition/
├── models/ - Model architecture related files
├── preprocessing/ - Data preprocessing related files
├── utils/ - Utility functions
├── tests/ - Test files
├── config/ - Configuration files
├── visualizations/ - Visualization results
├── train.py - Training script
├── evaluate.py - Evaluation script
├── inference.py - Inference script
└── requirements.txt - Dependencies
```

## Installation

```bash
# Clone the repository
git clone https://github.com/495008566/5500.git
cd 5500

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training

```bash
python train.py --config config/train_config.yaml
```

### Evaluation

```bash
python evaluate.py --config config/eval_config.yaml
```

### Inference

```bash
python inference.py --input <input_video> --output <output_path>
```

## Experimental Results

The system achieves state-of-the-art performance on cross-view gait recognition tasks:

- Average Rank-1 accuracy: 96.0% (NM), 87.3% (BG), 85.8% (CL)
- Processing time: 0.18 seconds per person

## License

This project is licensed under the MIT License - see the LICENSE file for details.
