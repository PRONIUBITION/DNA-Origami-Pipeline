import sys, os
from pathlib import Path
import numpy as np
import torch
import cv2

script_dir = Path(__file__).resolve().parent
monomer_root = script_dir.parent
repo_root = monomer_root.parent

yolo_dir = Path(os.environ.get("YOLOV5_DIR", repo_root / "third_party" / "yolov5")).resolve()
yolo_weights = Path(os.environ.get("YOLO_WEIGHTS", repo_root / "models" / "yolo_best.pt")).resolve()
test_dir = Path(os.environ.get("TEST_DIR", repo_root / "data" / "processed" / "test")).resolve()
out_seg_dir = Path(os.environ.get("OUTPUT_DIR", repo_root / "outputs" / "segmentation")).resolve()
yolo_conf = 0.78
yolo_iou = 0.10
box_expand = 0.15
dilation_pct = 0.20

out_seg_dir.mkdir(parents=True, exist_ok=True)

print("Loading YOLOv5...")
sys.path.insert(0, str(yolo_dir))
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes, check_img_size
from utils.augmentations import letterbox
from utils.segment.general import process_mask

device = torch.device(
    "cuda" if torch.cuda.is_available() and os.environ.get("DEVICE", "cuda") != "cpu" else "cpu"
)
model_yolo = DetectMultiBackend(
    yolo_weights, device=device, dnn=False, data=str(yolo_dir / "data" / "coco128.yaml"), fp16=False
)
model_yolo.model.float()
model_yolo.model.eval()
stride = model_yolo.stride
imgsz = check_img_size(480, s=stride)
print("YOLOv5 loaded OK")

crop_idx = 0
for fname in os.listdir(str(test_dir)):
    if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".tif")):
        continue
    print("\n--- Processing {} ---".format(fname))
    raw_data = np.fromfile(str(test_dir / fname), dtype=np.uint8)
    img0 = cv2.imdecode(raw_data, cv2.IMREAD_COLOR)
    if img0 is None:
        print("  ERROR: cannot read")
        continue
    orig_h, orig_w = img0.shape[:2]
    print("  Size: {}x{}".format(orig_w, orig_h))

    im_lb, ratio, (dw, dh) = letterbox(img0, imgsz, stride=stride, auto=True)
    lb_h, lb_w = im_lb.shape[:2]
    im = im_lb.transpose((2, 0, 1))[::-1]
    im = np.ascontiguousarray(im)
    im = torch.from_numpy(im).to(device).float() / 255.0
    if im.ndim == 3:
        im = im[None]

    pred = model_yolo(im)
    if isinstance(pred, (list, tuple)):
        pred_det = pred[0]
        p = pred[1] if len(pred) > 1 else None
    else:
        pred_det = pred
        p = None

    det = non_max_suppression(pred_det, yolo_conf, yolo_iou, None, False, max_det=1000, nm=32)[0]
    if det is None or len(det) == 0:
        print("  No detections")
        continue

    if p is not None and len(det):
        masks = process_mask(p[0], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)
        masks = (masks >= 0.5).float()
        boxes_lb = det[:, :4].clone()
        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], img0.shape).round()
    else:
        masks = None

    print("  {} monomers detected".format(len(det)))
    for i, d in enumerate(det):
        x1, y1, x2, y2 = int(d[0]), int(d[1]), int(d[2]), int(d[3])
        w, h = x2 - x1, y2 - y1
        x1 = max(0, int(x1 - w * box_expand))
        y1 = max(0, int(y1 - h * box_expand))
        x2 = min(orig_w, int(x2 + w * box_expand))
        y2 = min(orig_h, int(y2 + h * box_expand))

        if masks is not None:
            x1_lb, y1_lb, x2_lb, y2_lb = boxes_lb[i].int().tolist()
            w_lb = x2_lb - x1_lb
            h_lb = y2_lb - y1_lb
            x1_lb = max(0, int(x1_lb - w_lb * box_expand))
            y1_lb = max(0, int(y1_lb - h_lb * box_expand))
            x2_lb = min(lb_w, int(x2_lb + w_lb * box_expand))
            y2_lb = min(lb_h, int(y2_lb + h_lb * box_expand))

            mask_single = masks[i].cpu().numpy().astype(np.uint8) * 255
            mask_cropped_lb = mask_single[y1_lb:y2_lb, x1_lb:x2_lb]

            crop_w = x2 - x1
            crop_h = y2 - y1
            mask_orig = cv2.resize(
                mask_cropped_lb, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST
            )

            k = max(1, int(dilation_pct * min(w, h)))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            mask_dilated = cv2.dilate(mask_orig, kernel, iterations=1)

            crop_masked = cv2.bitwise_and(img0[y1:y2, x1:x2], img0[y1:y2, x1:x2], mask=mask_dilated)
            crop_masked = cv2.cvtColor(crop_masked, cv2.COLOR_BGR2GRAY)
        else:
            crop_masked = img0[y1:y2, x1:x2].copy()
            crop_masked = cv2.cvtColor(crop_masked, cv2.COLOR_BGR2GRAY)

        crop_name = "crop_{:03d}.png".format(crop_idx)
        cv2.imencode(".png", crop_masked)[1].tofile(str(out_seg_dir / crop_name))
        crop_idx += 1

print("\nDone. {} crops saved to {}".format(crop_idx, out_seg_dir))
