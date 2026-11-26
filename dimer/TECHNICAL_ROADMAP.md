# Dimer Technical Roadmap

## Objective

Detect dimers in AFM images, localize the white-dot keypoints, and convert
their geometry into a two-bit structural code.

## Architecture

```text
AFM image
  -> grayscale and image preparation
  -> YOLOv5-Seg instance segmentation
  -> expanded bounding box and dilated mask
  -> masked grayscale instance
  -> HRNet keypoint heatmaps
  -> peak extraction and grouping
  -> two-bit code or invalid
```

## Stage 1: Instance Segmentation

The segmentation stage uses YOLOv5-Seg with three classes:

| Class | Meaning |
| --- | --- |
| `dimmer` | Complete dimer |
| `monomer` | Monomer or incomplete fragment |
| `invalid` | Structurally indeterminate instance |

The stage-one configuration is stored in `stage1_yolo/configs`. Instance masks
are dilated before cropping to retain white dots close to the structure edge.

## Stage 2: Keypoint Detection

The HRNet stage predicts four heatmaps:

- `left_dot_1`
- `left_dot_2`
- `right_dot_1`
- `right_dot_2`

Peaks are decoded with a confidence threshold and non-maximum suppression. The
distance between the two points on each side determines one bit. Missing or
low-confidence points produce an `invalid` result.

## Evaluation

Use source-image-level train, validation, and test splits. Report:

- YOLO precision, recall, F1, box mAP, and mask mAP;
- HRNet PCK and OKS;
- code-level confusion matrices;
- invalid-instance rates.

Training-derived test sets may be used for debugging but must not be reported
as independent test performance.
