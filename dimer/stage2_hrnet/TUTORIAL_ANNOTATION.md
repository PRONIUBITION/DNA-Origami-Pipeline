# Keypoint Annotation Tutorial

This guide describes how to annotate white-dot keypoints with Labelme for the
dimer and monomer HRNet models.

## Installation

```bash
python -m pip install labelme
```

Launch Labelme:

```bash
labelme
```

## Data Layout

Keep source images and annotations separate:

```text
data/
├── processed/
│   ├── train/
│   └── val/
└── annotations/
    ├── train/
    └── val/
```

Each image must have a same-stem JSON annotation.

## Label Definitions

For dimer/four-point tasks:

```text
left_dot_1
left_dot_2
right_dot_1
right_dot_2
```

For three-point monomer variants:

```text
dot_1
dot_2
dot_3
```

For two-point monomer variants:

```text
dot_1
dot_2
```

Do not encode missing points with arbitrary coordinates. Omit the point or set
visibility to zero.

## Annotation Procedure

1. Open the processed image directory in Labelme.
2. Select `Create Point`.
3. Place the point at the center of the visible white dot.
4. Assign the exact configured keypoint label.
5. Save the JSON beside the corresponding annotation directory.
6. Repeat every image without carrying labels between images.

## Quality Control

Check that:

- every labeled image has a matching JSON file;
- no keypoint name is duplicated within an instance;
- missing points are not assigned invented positions;
- closely spaced dots remain distinguishable;
- points remain inside the structure crop;
- the required number of labels matches the selected class configuration.

## Common Problems

### A point is slightly offset

Use a consistent center convention for all images. Small random annotation
offsets increase label noise and reduce PCK/OKS.

### A dot is not clearly visible

Do not infer a missing dot from the expected code. Omit the keypoint and retain
the image as a negative or invalid example.

### Label names differ from the configuration

Update the annotation or configuration before training. Silent renaming can
shift heatmap channels and invalidate the model output.

### The same image appears in train and validation

Re-split by source image. Augmented copies of one source must stay in one
split.
