# Dimer YOLOv5 Training Record

## Experiment

| Field | Value |
| --- | --- |
| Model | YOLOv5s-Seg |
| Input mode | Grayscale AFM images |
| Run | `gray_06` |
| Start time | 2026-06-25 14:26 |
| Dataset size | 728 training images / 112 validation images |
| Epochs | 500 |
| Batch size | 8 |
| Early-stopping patience | 50 |

## Main Changes

- Moderate geometric augmentation;
- cosine learning-rate schedule;
- no frozen layers;
- longer early-stopping patience.

## Monitoring

Training produces:

- `results.csv`;
- loss and metric plots;
- validation visualizations;
- TensorBoard event files;
- best and last model checkpoints.

Use `results.csv` to identify convergence and select checkpoints according to
the preregistered validation metric.
