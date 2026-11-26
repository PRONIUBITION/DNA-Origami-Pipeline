import csv
import os
import torch
from pathlib import Path

print("=" * 70)
print("""Best metrics for each YOLOv5-seg training run (sorted by Mask mAP@50)""")
print("=" * 70)

DIMER_ROOT = Path(__file__).resolve().parents[1]
runs = Path(os.environ.get("YOLO_RUNS", DIMER_ROOT.parent / "outputs" / "yolo_runs" / "train-seg"))
results = []
for run_dir in sorted(runs.iterdir()):
    if not run_dir.is_dir():
        continue
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        continue
    try:
        with open(csv_path, "r") as f:
            reader = list(csv.reader(f))
        if len(reader) < 2:
            continue
        header = [h.strip() for h in reader[0]]
        map50_idx = None
        for i, h in enumerate(header):
            if "metrics/mAP_0.5(M)" in h:
                map50_idx = i
                break
        if map50_idx is None:
            continue
        best_val = -1
        best_epoch = None
        best_row = None
        for row in reader[1:]:
            if len(row) <= map50_idx:
                continue
            try:
                val = float(row[map50_idx].strip())
                if val > best_val:
                    best_val = val
                    best_epoch = row[0].strip()
                    best_row = row
            except:
                continue
        if best_row:

            def get(name):
                for i, h in enumerate(header):
                    if name in h and i < len(best_row):
                        return float(best_row[i].strip())
                return None

            results.append(
                {
                    "run": run_dir.name,
                    "epoch": best_epoch,
                    "mAP50_mask": get("metrics/mAP_0.5(M)"),
                    "mAP5095_mask": get("metrics/mAP_0.5:0.95(M)"),
                    "precision_mask": get("metrics/precision(M)"),
                    "recall_mask": get("metrics/recall(M)"),
                    "val_seg_loss": get("val/seg_loss"),
                }
            )
    except Exception as e:
        pass

results.sort(key=lambda x: x["mAP50_mask"] or 0, reverse=True)
print(f"""{len(results)} valid training runs found. Top 5: \n""")
for r in results[:5]:
    print(f"Run: {r['run']}")
    print(f"""Best Epoch: {r["epoch"]}""")
    print(f"  Mask mAP@50:    {r['mAP50_mask']:.4f}" if r["mAP50_mask"] else "  Mask mAP@50: N/A")
    print(
        f"  Mask mAP@50-95: {r['mAP5095_mask']:.4f}"
        if r["mAP5095_mask"]
        else "  Mask mAP@50-95: N/A"
    )
    print(
        f"  Mask Precision: {r['precision_mask']:.4f}"
        if r["precision_mask"]
        else "  Mask Precision: N/A"
    )
    print(f"  Mask Recall:    {r['recall_mask']:.4f}" if r["recall_mask"] else "  Mask Recall: N/A")
    print(
        f"  Val Seg Loss:   {r['val_seg_loss']:.4f}" if r["val_seg_loss"] else "  Val Seg Loss: N/A"
    )
    print()

print("=" * 70)
print("""Best metrics for each HRNet training run (sorted by validation loss)""")
print("=" * 70)

hrnet_runs = Path(os.environ.get("HRNET_RUNS", DIMER_ROOT / "stage2_hrnet" / "runs"))
checkpoints = []
for run_dir in sorted(hrnet_runs.iterdir()):
    if not run_dir.is_dir() or not run_dir.name.startswith("train_"):
        continue
    for weight_file in run_dir.glob("weights/*.pth"):
        try:
            ckpt = torch.load(weight_file, map_location="cpu")
            val_loss = ckpt.get("val_loss", None)
            epoch = ckpt.get("epoch", None)
            checkpoints.append(
                {
                    "run": run_dir.name,
                    "file": weight_file.name,
                    "epoch": epoch,
                    "val_loss": val_loss,
                }
            )
        except:
            pass

checkpoints.sort(key=lambda x: x["val_loss"] if x["val_loss"] is not None else float("inf"))
print(f"""{len(checkpoints)} checkpoints found. Top 5: \n""")
for c in checkpoints[:5]:
    print(f"{c['run']}/{c['file']}")
    print(
        f"  Epoch: {c['epoch']}, Val Loss: {c['val_loss']:.6f}"
        if c["val_loss"]
        else f"  Epoch: {c['epoch']}, Val Loss: N/A"
    )
    print()
