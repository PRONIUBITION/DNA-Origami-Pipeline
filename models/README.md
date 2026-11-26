# Model Weights

Model weights are intentionally excluded from Git history.

Expected external files:

```text
yolo_best.pt
hrnet_best.pth
```

For YOLOv5, first install the official repository at a pinned commit and then
set `YOLOV5_DIR` and `YOLO_WEIGHTS`. For HRNet, use the weight file matching
the configuration stored in `dimer/stage2_hrnet/configs` or
`monomer/stage2_hrnet/configs`.

Record checksums when publishing weights:

```bash
sha256sum yolo_best.pt hrnet_best.pth
```
