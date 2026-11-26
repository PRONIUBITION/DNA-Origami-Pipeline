# Data Layout

Raw AFM images, annotations, processed crops, and model weights are excluded
from Git because they are large and may be subject to research-data governance.

Use environment variables to point the pipeline at an external data directory:

```text
DNA_DATA_ROOT=/path/to/dnaorigami-data
DNA_OUTPUT_ROOT=/path/to/dnaorigami-outputs
YOLOV5_DIR=/path/to/yolov5
YOLO_WEIGHTS=/path/to/yolov5_best.pt
HRNET_WEIGHTS=/path/to/hrnet_best.pth
```

Recommended external layout:

```text
dnaorigami-data/
├── raw/
├── annotations/
├── processed/
├── yolo_gray/
├── multicodes/
│   ├── chains/
│   ├── templates/
│   └── crops/
└── test/
```

For YOLO segmentation, each image must have a matching normalized polygon
label. LabelMe JSON conversion and train/validation splitting are implemented
in the stage-one utilities.
