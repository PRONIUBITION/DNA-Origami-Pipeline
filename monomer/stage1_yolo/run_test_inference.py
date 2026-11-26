import os, re, json, subprocess, numpy as np, cv2
from pathlib import Path
from datetime import datetime
from settings import YOLOV5_DIR, YOLO_WEIGHTS, data_path, output_path

PYTHON = os.sys.executable
WEIGHTS = YOLO_WEIGHTS
TEST_DIR = Path(os.environ.get("TEST_DIR", data_path("test"))).resolve()
OUT_BASE = Path(os.environ.get("OUTPUT_DIR", output_path("predictions"))).resolve()
CONF_THRES = 0.45
IOU_THRES = 0.15
MARGIN = 0.10


def imread_any(path):
    with open(path, "rb") as f:
        return cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)


def imwrite_any(path, img):
    ext = os.path.splitext(path)[1] or ".png"
    ret, buf = cv2.imencode(ext, img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if ret:
        with open(path, "wb") as f:
            f.write(buf.tobytes())
        return True
    return False


def compute_iou(b1, b2):
    x1, y1, x2, y2 = b1
    x3, y3, x4, y4 = b2
    xi1 = max(x1, x3)
    yi1 = max(y1, y3)
    xi2 = min(x2, x4)
    yi2 = min(y2, y4)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    a1 = (x2 - x1) * (y2 - y1)
    a2 = (x4 - x3) * (y4 - y3)
    return inter / (a1 + a2 - inter + 1e-6) if a1 + a2 - inter > 0 else 0


stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_name = f"test_{stamp}"
cmd = [
    str(PYTHON),
    str(YOLOV5_DIR / "segment" / "predict.py"),
    f"--weights={WEIGHTS}",
    f"--source={TEST_DIR}",
    "--imgsz=480",
    "--conf-thres=0.25",
    "--iou-thres=0.45",
    "--save-txt",
    "--save-conf",
    "--exist-ok",
    f"--project={YOLOV5_DIR / 'runs' / 'predict-seg'}",
    f"--name={run_name}",
]
subprocess.run(cmd, capture_output=True, cwd=str(YOLOV5_DIR))

pred_dir = YOLOV5_DIR / "runs" / "predict-seg" / run_name
labels_dir = pred_dir / "labels"
out_dir = OUT_BASE / run_name
vis_dir = out_dir / "visualizations"
vis_dir.mkdir(parents=True, exist_ok=True)
stats = {}
total_det = 0

for entry in os.scandir(str(TEST_DIR)):
    if not entry.name.lower().endswith((".png", ".jpg")):
        continue
    stem = entry.name.rsplit(".", 1)[0]
    lab = labels_dir / f"{stem}.txt"
    img = imread_any(entry.path)
    if img is None:
        print(f"""[!] : {stem}""")
        stats[stem] = {"error": "cannot read"}
        continue
    h, w = img.shape[:2]
    vis = img.copy()
    dets = []
    if lab.exists():
        with open(lab) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 7:
                    continue
                conf = float(parts[1])
                pts = np.array([float(v) for v in parts[2:]]).reshape(-1, 2)
                xn1 = float(pts[:, 0].min())
                yn1 = float(pts[:, 1].min())
                xn2 = float(pts[:, 0].max())
                yn2 = float(pts[:, 1].max())
                dets.append({"conf": conf, "bbox": (xn1, yn1, xn2, yn2)})
    dets = [d for d in dets if d["conf"] >= CONF_THRES]
    dets.sort(key=lambda d: d["conf"], reverse=True)
    kept = []
    for d in dets:
        if not any(compute_iou(d["bbox"], k["bbox"]) > IOU_THRES for k in kept):
            kept.append(d)
    confs = [d["conf"] for d in kept]
    stats[stem] = {
        "raw": sum(1 for _ in open(lab)) if lab.exists() else 0,
        "kept": len(kept),
        "confs": [round(c, 4) for c in confs],
    }
    total_det += len(kept)
    print(f"  {stem}: {stats[stem]['raw']} raw -> {len(kept)} kept")
    for d in kept:
        xn1, yn1, xn2, yn2 = d["bbox"]
        bw, bh = xn2 - xn1, yn2 - yn1
        mx = bw * MARGIN
        my = bh * MARGIN
        x1 = int(max(0, (xn1 - mx)) * w)
        y1 = int(max(0, (yn1 - my)) * h)
        x2 = int(min(1, (xn2 + mx)) * w)
        y2 = int(min(1, (yn2 + my)) * h)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 0), 2)
        label = f"mon {d['conf']:.2f}"
        fs = 0.35
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        cv2.rectangle(vis, (x1, y1 - th - 2), (x1 + tw, y1), (0, 200, 0), -1)
        cv2.putText(
            vis, label, (x1, y1 - 1), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 1, cv2.LINE_AA
        )
    safe_stem = re.sub(r"[^\w\s-]", "_", stem)
    imwrite_any(str(vis_dir / f"{safe_stem}.png"), vis)

with open(out_dir / "stats.json", "w", encoding="utf-8") as f:
    json.dump(
        {
            "model": str(WEIGHTS),
            "config": {"conf_thres": CONF_THRES, "iou_thres_nms": IOU_THRES, "margin": MARGIN},
            "results": stats,
            "total_detections": total_det,
        },
        f,
        indent=2,
        ensure_ascii=False,
    )
print(f"""\n : {total_det}""")
print(f""": {vis_dir}""")
