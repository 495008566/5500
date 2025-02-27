# Dataset Download Instructions

## CASIA-B Dataset

The CASIA-B dataset is a large multi-view gait database created by the Institute of Automation, Chinese Academy of Sciences. It contains 124 subjects captured from 11 views.

### Download Steps

1. Visit the official website: http://www.cbsr.ia.ac.cn/english/Gait%20Databases.asp
2. Register for an account and submit an application for access
3. Once approved, download the dataset
4. Extract the dataset to the `data/casia_b` directory

### Dataset Structure

The dataset should have the following structure:
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

### Dataset Details

- 124 subjects (person IDs from 001 to 124)
- 11 view angles (0°, 18°, 36°, 54°, 72°, 90°, 108°, 126°, 144°, 162°, 180°)
- 10 sequences per subject (6 normal walking, 2 with bag, 2 with coat)
- Resolution: 320×240 pixels
