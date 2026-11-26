import sys, os, time, json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import cv2
import importlib.util as iu

script_dir = Path(__file__).resolve().parent
monomer_root = script_dir.parent
repo_root = monomer_root.parent
stage2_root = monomer_root / "stage2_hrnet"
MONOMER_TYPE = os.environ.get("MONOMER_TYPE", "0")
if MONOMER_TYPE not in {"0", "1"}:
    raise SystemExit("pipeline_two_point.py supports MONOMER_TYPE=0 or 1")

# --- Paths ---
yolo_dir = Path(os.environ.get("YOLOV5_DIR", repo_root / "third_party" / "yolov5")).resolve()
yolo_weights = Path(os.environ.get("YOLO_WEIGHTS", repo_root / "models" / "yolo_best.pt")).resolve()
test_dir = Path(os.environ.get("TEST_DIR", repo_root / "data" / "processed" / "test")).resolve()
hrnet_ckpt = Path(os.environ.get("HRNET_WEIGHTS", repo_root / "models" / "hrnet_best.pt")).resolve()
yolo_conf = 0.78
yolo_iou = 0.10
box_expand = 0.15
hr_input_size = (256, 256)
hr_hm_size = (64, 64)
hr_th = 0.05
dilation_pct = 0.20
try:
    font_label = ImageFont.truetype(os.environ.get("FONT_PATH", "arial.ttf"), 32)
except OSError:
    font_label = ImageFont.load_default()

# --- Load HRNet ---
print("Loading HRNet...")
spec = iu.spec_from_file_location("hrnet_mod", str(stage2_root / "models" / "hrnet.py"))
hrnet_mod = iu.module_from_spec(spec)
spec.loader.exec_module(hrnet_mod)
SimpleHRNet = hrnet_mod.SimpleHRNet
spec2 = iu.spec_from_file_location("utils_heatmap", str(stage2_root / "utils" / "heatmap.py"))
utils_mod = iu.module_from_spec(spec2)
spec2.loader.exec_module(utils_mod)
heatmap_to_coordinates = utils_mod.heatmap_to_coordinates
device = torch.device(
    "cuda" if torch.cuda.is_available() and os.environ.get("DEVICE", "cuda") != "cpu" else "cpu"
)
ckpt = torch.load(hrnet_ckpt, map_location=device)
model_hr = SimpleHRNet(num_keypoints=2, in_channels=1).to(device)
model_hr.load_state_dict(ckpt.get("model_state_dict", ckpt.get("model")))
model_hr.eval()
hrnet_epoch = ckpt.get("epoch", "?")
print("HRNet loaded OK (epoch {})".format(hrnet_epoch))

# --- Load YOLOv5 ---
print("Loading YOLOv5...")
sys.path.insert(0, str(yolo_dir))
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes, check_img_size
from utils.augmentations import letterbox
from utils.segment.general import process_mask

model_yolo = DetectMultiBackend(
    yolo_weights, device=device, dnn=False, data=str(yolo_dir / "data" / "coco128.yaml"), fp16=False
)
model_yolo.model.float()
model_yolo.model.eval()
stride = model_yolo.stride
imgsz = check_img_size(480, s=stride)
print("YOLOv5 loaded OK")

# --- Output ---
timestamp = time.strftime("%Y%m%d_%H%M%S")
out_dir = Path(
    os.environ.get("OUTPUT_DIR", repo_root / "outputs" / "monomer")
).resolve() / "pipeline_type_{}_{}".format(MONOMER_TYPE, timestamp)
out_dir.mkdir(parents=True, exist_ok=True)
crops_dir = out_dir / "crops_masked"
crops_dir.mkdir(exist_ok=True)
vis_dir = out_dir / "visualizations"
hrnet_dir = out_dir / "hrnet_results"
hrnet_dir.mkdir(exist_ok=True)
vis_dir.mkdir(exist_ok=True)

# --- Process each image ---
all_results = []
total_crops_0 = 0
total_crops_invalid = 0

# Use listdir with imdecode for CJK path support
test_files = []
for fname in os.listdir(str(test_dir)):
    if fname.lower().endswith((".png", ".jpg", ".jpeg", ".tif")):
        test_files.append(fname)

for img_name in test_files:
    print("\n--- Processing {} ---".format(img_name))
    # Read image with CJK path support
    raw_data = np.fromfile(str(test_dir / img_name), dtype=np.uint8)
    img0 = cv2.imdecode(raw_data, cv2.IMREAD_COLOR)
    if img0 is None:
        print("  ERROR: cannot read")
        continue

    orig_h, orig_w = img0.shape[:2]
    print("  Image size: {}x{}".format(orig_w, orig_h))

    # YOLO preprocess
    im_lb, ratio, (dw, dh) = letterbox(img0, imgsz, stride=stride, auto=True)
    im = im_lb
    im = im.transpose((2, 0, 1))[::-1]
    im = np.ascontiguousarray(im)
    im = torch.from_numpy(im).to(device).float() / 255.0
    if im.ndim == 3:
        im = im[None]

    # YOLO inference
    with torch.no_grad():
        pred = model_yolo(im, augment=False, visualize=False)
        if isinstance(pred, (list, tuple)):
            pred_det, pred_proto = pred[0], pred[1]
        else:
            pred_det = pred
        det = non_max_suppression(pred_det, yolo_conf, yolo_iou, None, False, max_det=1000, nm=32)[
            0
        ]
        # Get masks and map to original image
        masks_lb = process_mask(pred_proto[0], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)
        hc = int(orig_h * ratio[0] + 0.5)
        wc = int(orig_w * ratio[1] + 0.5)
        ph, pw = int(dh + 0.5), int(dw + 0.5)
        mn = masks_lb.cpu().numpy()
        mo = np.zeros((len(mn), orig_h, orig_w), np.float32)
        for mi in range(len(mn)):
            mc = mn[mi, ph : ph + hc, pw : pw + wc]
            mo[mi] = cv2.resize(mc, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    if det is None or len(det) == 0:
        continue
    # Get masks and map to original image

    # Scale boxes to original image
    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], img0.shape).round()

    # Expand boxes
    exp_x = (det[:, 2] - det[:, 0]) * box_expand
    exp_y = (det[:, 3] - det[:, 1]) * box_expand
    det[:, 0] = (det[:, 0] - exp_x).clamp(0)
    det[:, 1] = (det[:, 1] - exp_y).clamp(0)
    det[:, 2] = (det[:, 2] + exp_x).clamp(0, orig_w)
    det[:, 3] = (det[:, 3] + exp_y).clamp(0, orig_h)

    print("  {} monomers detected".format(len(det)))

    # Draw full image visualization (NO numbering)
    vis_full = img0.copy()
    box_color = (0, 255, 0)  # green boxes, no numbers

    img_results = []
    img_0_count = 0
    img_invalid_count = 0

    for i, d in enumerate(det):
        x1, y1, x2, y2 = int(d[0]), int(d[1]), int(d[2]), int(d[3])
        conf = float(d[4])

        # Crop monomer
        crop = img0[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        # Draw box on full image (no index number)
        cv2.rectangle(vis_full, (x1, y1), (x2, y2), box_color, 1)
        # Draw confidence score above box
        conf_text = "{:.2f}".format(conf)
        cv2.putText(vis_full, conf_text, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, box_color, 1)

        # Save crop
        safe_name = img_name.replace("(", "_").replace(")", "_").replace(" ", "_")
        crop_name = "{}_crop_{:03d}.png".format(Path(img_name).stem, i)
        crop_path = str(crops_dir / crop_name)

        # Convert to grayscale, resize to 256x256
        crop_h, crop_w = crop.shape[:2]
        mask_raw = (mo[i, y1:y2, x1:x2] > 0.5).astype(np.uint8)
        ks = max(3, int(min(crop_h, crop_w) * dilation_pct) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
        mask_dil = cv2.dilate(mask_raw, kernel, iterations=1)
        crop_masked = crop.copy()
        for ci in range(3):
            crop_masked[:, :, ci] *= mask_dil
        crop_gray = cv2.cvtColor(crop_masked, cv2.COLOR_BGR2GRAY)
        crop_resized = cv2.resize(crop_gray, hr_input_size, interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(crop_path, crop_resized)
        hrnet_crop_path = str(hrnet_dir / ("hrnet_" + crop_name))

        # HRNet preprocessing
        img_tensor = (
            torch.from_numpy(((crop_resized / 127.5) - 1.0)[np.newaxis, np.newaxis, ...])
            .float()
            .to(device)
        )

        # HRNet inference
        with torch.no_grad():
            pred_hm = model_hr(img_tensor).cpu().numpy()[0]

        # Extract keypoints
        coords = heatmap_to_coordinates(pred_hm, hr_th)

        # Count valid keypoints
        valid_kps = [c for c in coords if c[2] > 0 and c[0] >= 0 and c[1] >= 0]
        num_kp = len(valid_kps)
        label_text = MONOMER_TYPE if num_kp == 2 else "invalid"
        label_color = (0, 200, 0) if num_kp == 2 else (0, 0, 200)
        label_color_text = (0, 0, 0)  # label font color black

        # Draw on crop visualization
        crop_vis = cv2.cvtColor(crop_resized, cv2.COLOR_GRAY2BGR)
        dot_colors = [(0, 0, 255), (0, 255, 0)]
        dot_labels = ["dot_1", "dot_2"]
        kp_info = []

        for ch, (x, y, conf) in enumerate(coords[:2]):
            if conf > 0 and x >= 0 and y >= 0:
                cx = int(x * hr_input_size[0] / hr_hm_size[0])
                cy = int(y * hr_input_size[1] / hr_hm_size[1])
                cv2.circle(crop_vis, (cx, cy), 4, dot_colors[ch], -1)
                kp_info.append(
                    {
                        "channel": ch,
                        "label": dot_labels[ch],
                        "x_heatmap": float(x),
                        "y_heatmap": float(y),
                        "confidence": float(conf),
                    }
                )

        h_display, w_display = crop_vis.shape[:2]
        # Add space at bottom for label
        label_h = 52
        vis_with_label = np.full((h_display + label_h, w_display, 3), 255, dtype=np.uint8)
        vis_with_label[:h_display, :, :] = crop_vis
        # Render label with Arial TrueType font via PIL
        vis_pil = Image.fromarray(vis_with_label)
        draw = ImageDraw.Draw(vis_pil)
        bbox = draw.textbbox((0, 0), label_text, font=font_label)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        tx = (w_display - tw) // 2
        ty = h_display + (label_h - th) // 2
        draw.text((tx, ty), label_text, font=font_label, fill=label_color_text)
        vis_with_label = np.array(vis_pil)
        cv2.imwrite(hrnet_crop_path, vis_with_label)

        if num_kp == 2:
            img_0_count += 1
            total_crops_0 += 1
        else:
            img_invalid_count += 1
            total_crops_invalid += 1

        img_results.append(
            {
                "crop_index": i,
                "crop_file": crop_name,
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "confidence": conf,
                "num_keypoints": num_kp,
                "label": label_text,
                "keypoints": kp_info,
            }
        )

    # Save full image visualization
    full_vis_name = Path(img_name).stem + ".png"
    full_vis_path = str(vis_dir / full_vis_name)
    cv2.imwrite(full_vis_path, vis_full)

    all_results.append(
        {
            "image": img_name,
            "size": [orig_w, orig_h],
            "num_monomers": len(img_results),
            "count_0": img_0_count,
            "count_invalid": img_invalid_count,
            "monomers": img_results,
        }
    )

    print("  Result: 0={}, invalid={}".format(img_0_count, img_invalid_count))

# --- Summary ---
summary = {
    "pipeline": "YOLOv5-seg + SimpleHRNet keypoint detection",
    "yolo_model": str(yolo_weights),
    "yolo_conf_threshold": yolo_conf,
    "hrnet_model": str(hrnet_ckpt),
    "hrnet_epoch": hrnet_epoch,
    "total_images": len(all_results),
    "total_monomers": total_crops_0 + total_crops_invalid,
    "type": MONOMER_TYPE,
    "total_valid": total_crops_0,
    "total_invalid": total_crops_invalid,
    "results": all_results,
}

print("\n" + "=" * 50)
print("PIPELINE COMPLETE")
print("=" * 50)
print("Total images: {}".format(len(all_results)))
print("Total monomers: {}".format(total_crops_0 + total_crops_invalid))
print("  {} (2 keypoints): {}".format(MONOMER_TYPE, total_crops_0))
print("  invalid (<2 keypoints): {}".format(total_crops_invalid))
print("Results saved to: {}".format(out_dir))
print("=" * 50)

# Save JSON
with open(out_dir / "pipeline_results.json", "w") as f:

    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, (np.ndarray,)):
                return obj.tolist()
            return super().default(obj)

    json.dump(summary, f, indent=2, ensure_ascii=False, cls=NpEncoder)

# Save summary text
with open(out_dir / "pipeline_summary.txt", "w", encoding="utf-8") as f:
    f.write("=== Pipeline Results ===\n")
    f.write("Timestamp: {}\n".format(timestamp))
    f.write("YOLO weights: {}\n".format(yolo_weights))
    f.write("YOLO conf_thres: {:.2f}\n".format(yolo_conf))
    f.write("HRNet weights: {}\n".format(hrnet_ckpt))
    f.write("HRNet epoch: {}\n".format(hrnet_epoch))
    f.write("\n--- Statistics ---\n")
    f.write("Total monomers: {:3d}\n".format(total_crops_0 + total_crops_invalid))
    f.write("  {} (2 kp):    {:3d}\n".format(MONOMER_TYPE, total_crops_0))
    f.write("  invalid:      {:3d}\n".format(total_crops_invalid))
    f.write("\n--- Per Image ---\n")
    for r in all_results:
        f.write("\n{}:\n".format(r["image"]))
        f.write("  Size: {}x{}\n".format(r["size"][0], r["size"][1]))
        f.write(
            "  Monomers: {} (0={}, invalid={})\n".format(
                r["num_monomers"], r["count_0"], r["count_invalid"]
            )
        )
        for m in r["monomers"]:
            kp_str = ", ".join(
                [
                    "{}:({:.0f},{:.0f})".format(k["label"], k["x_heatmap"], k["y_heatmap"])
                    for k in m["keypoints"]
                ]
            )
            f.write(
                "  Crop {:02d} [conf={:.2f}] {} {}\n".format(
                    m["crop_index"], m["confidence"], m["label"], kp_str
                )
            )
