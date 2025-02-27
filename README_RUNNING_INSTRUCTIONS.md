# Running Instructions

## Dataset Preparation
1. Download the CASIA-B dataset and place it in the `data/casia_b` directory
2. The dataset should have the following structure:
   ```
   data/casia_b/
   ├── 001/
   │   ├── nm-01/
   │   ├── nm-02/
   │   └── ...
   ├── 002/
   │   ├── nm-01/
   │   ├── nm-02/
   │   └── ...
   └── ...
   ```

## Training
To train the model, run:
```bash
python train_real_datasets.py --dataset casia_b --checkpoint_dir checkpoints/casia_b --log_dir logs/casia_b
```

## Evaluation
To evaluate the model, run:
```bash
python evaluate.py --dataset casia_b --checkpoint_path checkpoints/casia_b/best_model.pth --output_dir results/casia_b
```

## Ablation Study
To run ablation studies, run:
```bash
python ablation_study.py --dataset casia_b --checkpoint_path checkpoints/casia_b/best_model.pth --output_dir results/ablation/casia_b --components all
```

## Training with Different Parameters
You can modify the configuration files in the `config` directory to change the model architecture and training parameters.
