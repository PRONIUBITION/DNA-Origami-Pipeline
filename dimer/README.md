# Dimer Workflow

This module detects dimer instances, predicts structural keypoints, and groups
the detected dots into a discrete code.

## Pipeline

```text
AFM image
  → grayscale and normalization
  → YOLOv5-Seg instance detection and mask generation
  → mask dilation and expanded crop
  → HRNet keypoint heatmaps
  → geometric grouping and code assignment
  → annotated image, JSON, and sequence text
```

## Requirements

- Python 3.9+
- PyTorch and torchvision
- OpenCV, NumPy, Pillow, PyYAML, tqdm
- Official YOLOv5 repository with the segmentation dependencies
- HRNet weight compatible with `stage2_hrnet/configs/config.yaml`

## Run

```bash
python dimer/pipeline.py \
  --input /path/to/images \
  --yolo-weights /path/to/yolo_best.pt \
  --hrnet-weights /path/to/hrnet_best.pth \
  --output /path/to/results
```

Set `YOLOV5_DIR` when YOLOv5 is installed outside
`third_party/yolov5`.

## Main Directories

- `stage1_yolo/`: LabelMe conversion, dataset preparation, segmentation,
  mask cropping, and inference.
- `stage2_hrnet/`: HRNet data preparation, training, prediction, grouping,
  and confusion-matrix evaluation.
- `datapredict/`: convenience runner for a complete two-stage prediction.
- `analysis/`: distance, grouping, confidence, and model-selection analyses.

## Outputs

Every run should be written outside the repository or under an ignored output
directory. Preserve `pipeline_results.json`, the resolved configuration, model
checksums, and the source image list for reproducibility.
