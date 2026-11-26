# Monomer Workflow

This module processes six monomer configurations: `0`, `1`, `00`, `01`, `10`,
and `11`.

- `0` and `1` require two valid keypoints.
- `00`, `01`, `10`, and `11` require three valid keypoints.

## Inference

```bash
python monomer/pipelines/pipeline.py \
  --type 00 \
  --input /path/to/images \
  --yolo-weights /path/to/yolo_best.pt \
  --hrnet-weights /path/to/type00.pth \
  --yolov5-dir /path/to/yolov5 \
  --output /path/to/results
```

The wrapper selects `pipeline_two_point.py` for `0` and `1`, and
`pipeline_three_point.py` for the other four types. This avoids mixing the
different keypoint-validation rules in a single branch-heavy function.

## Training

```bash
cd monomer/stage2_hrnet
python train.py --type 00
python predict.py --type 00 --weights /path/to/model.pth \
  --input /path/to/crops --output /path/to/results
```

The six configuration files are stored in
`stage2_hrnet/configs/types/type_<code>.yaml`.

## Stage-One Data

Stage-one utilities read from `DNA_DATA_ROOT` and write to `DNA_OUTPUT_ROOT`.
The default external layout is documented in the root `data/README.md`.

LabelMe conversion uses one class by default:

```bash
python monomer/stage1_yolo/labelme2yolo.py \
  --type 00 \
  --width 512 \
  --height 512 \
  --class-name monomer
```

## Validity Rule

The expected code is assigned only when the selected HRNet produces the
required number of accepted keypoints. A structure with missing, fused, or
low-confidence detections is labeled `invalid`; it is not silently promoted to
a valid code.
