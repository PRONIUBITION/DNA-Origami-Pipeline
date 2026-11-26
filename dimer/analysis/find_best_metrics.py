"""Extract the optimal training metrics for YOLOv5 and HRNet"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import re

print("=" * 70)
print("""YOLOv5-seg Training Metrics Summary""")
print("=" * 70)

DIMER_ROOT = Path(__file__).resolve().parents[1]
yolo_runs = Path(
    os.environ.get("YOLO_RUNS", DIMER_ROOT.parent / "outputs" / "yolo_runs" / "train-seg")
)
best_yolo = {}

for run_dir in sorted(yolo_runs.iterdir()):
    if not run_dir.is_dir():
        continue
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        continue

    try:
        df = pd.read_csv(csv_path)
        # The column name format in YOLOv5 results.csv is not fixed. Try to find the mAP column.
        cols = df.columns.tolist()

        # Find the column containing mAP
        map50_col = [c for c in cols if "metrics/mAP50" in c or "mAP50" in c]
        map5095_col = [c for c in cols if "metrics/mAP50-95" in c or "mAP50-95" in c]
        prec_col = [c for c in cols if "metrics/precision" in c or "precision" in c]
        rec_col = [c for c in cols if "metrics/recall" in c or "recall" in c]

        if map50_col:
            best_idx = df[map50_col[0]].idxmax()
            best_row = df.iloc[best_idx]

            metrics = {
                "run": run_dir.name,
                "epoch": int(best_row.iloc[0]) if pd.notna(best_row.iloc[0]) else "N/A",
                "mAP50": best_row[map50_col[0]] if map50_col else None,
                "mAP50-95": best_row[map5095_col[0]] if map5095_col else None,
                "precision": best_row[prec_col[0]] if prec_col else None,
                "recall": best_row[rec_col[0]] if rec_col else None,
            }
            best_yolo[run_dir.name] = metrics

    except Exception as e:
        print(f"""{run_dir.name}: Parsing failed - {e}""")

# Sort by mAP50
sorted_yolo = sorted(best_yolo.items(), key=lambda x: x[1]["mAP50"] or 0, reverse=True)
print(f"""\n {len(sorted_yolo)} valid training runs found. Top 5:""")
print("-" * 70)
for name, m in sorted_yolo[:5]:
    print(f"Run: {name}")
    print(f"""Best Epoch: {m["epoch"]}""")
    print(f"  mAP@50:    {m['mAP50']:.4f}" if m["mAP50"] else "  mAP@50: N/A")
    print(f"  mAP@50-95: {m['mAP50-95']:.4f}" if m["mAP50-95"] else "  mAP@50-95: N/A")
    print(f"  Precision: {m['precision']:.4f}" if m["precision"] else "  Precision: N/A")
    print(f"  Recall:    {m['recall']:.4f}" if m["recall"] else "  Recall: N/A")
    print()

# If best.pt exists, list it as well.
print("=" * 70)
print("""Summary of HRNet Training Metrics""")
print("=" * 70)

hrnet_runs = Path(os.environ.get("HRNET_RUNS", DIMER_ROOT / "stage2_hrnet" / "runs"))
best_hrnet = {}

for run_dir in sorted(hrnet_runs.iterdir()):
    if not run_dir.is_dir() or not run_dir.name.startswith("train_"):
        continue

    log_path = run_dir / "logs" / "train.log"
    if not log_path.exists():
        continue

    try:
        with open(log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        best_val_loss = float("inf")
        best_epoch = 0
        best_line = ""

        for line in lines:
            # Look for a format similar to "Epoch 55/100 - Val Loss: 0.00499"
            match = re.search(r"Epoch\s+(\d+).*?Val Loss:\s+([\d.]+)", line, re.IGNORECASE)
            if match:
                epoch = int(match.group(1))
                val_loss = float(match.group(2))
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    best_line = line.strip()

        # Also try to find the last line containing summary information.
        final_info = {}
        for line in lines[-20:]:
            if "best" in line.lower() or "summary" in line.lower():
                final_info["summary"] = line.strip()

        best_hrnet[run_dir.name] = {
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss if best_val_loss < float("inf") else None,
            "log_snippet": best_line,
        }

    except Exception as e:
        print(f"""{run_dir.name}: Parsing failed - {e}""")

# Sort by val_loss
sorted_hrnet = sorted(best_hrnet.items(), key=lambda x: x[1]["best_val_loss"] or float("inf"))
print(f"""\n Found {len(sorted_hrnet)} valid training runs, Top 5 (by validation loss):""")
print("-" * 70)
for name, m in sorted_hrnet[:5]:
    print(f"Run: {name}")
    print(f"""Best Epoch: {m["best_epoch"]}""")
    print(
        f"""Best Val Loss: {m["best_val_loss"]:.6f}"""
        if m["best_val_loss"]
        else """Best Val Loss: N/A"""
    )
    if m["log_snippet"]:
        print(f"""Log: {m["log_snippet"]}""")
    print()

# Additional: Check val_loss in the checkpoint
print("=" * 70)
print("""HRNet Checkpoint Validation Loss (Read Directly from Weight File)""")
print("=" * 70)
import torch

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
                    "path": str(weight_file),
                }
            )
        except:
            pass

checkpoints.sort(key=lambda x: x["val_loss"] if x["val_loss"] is not None else float("inf"))
print(f"""{len(checkpoints)} checkpoints found. Top 5:""")
print("-" * 70)
for c in checkpoints[:5]:
    print(f"{c['run']}/{c['file']}")
    print(
        f"  Epoch: {c['epoch']}, Val Loss: {c['val_loss']:.6f}"
        if c["val_loss"]
        else f"  Epoch: {c['epoch']}, Val Loss: N/A"
    )
    print()
