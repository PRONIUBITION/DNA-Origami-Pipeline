import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from settings import output_path

# ==================== Edit here ====================
LOG_DIR = output_path("yolo_runs", "train-seg")


# =================================================


def parse_results(log_file):
    """Parse results.csv"""
    epochs, box_loss, seg_loss, val_box_loss, val_seg_loss = [], [], [], [], []
    map50_box, map50_mask = [], []

    with open(log_file, "r") as f:
        lines = f.readlines()[1:]  # Skip header

    for line in lines:
        if not line.strip():
            continue
        parts = line.strip().split(",")
        if len(parts) < 20:
            continue

        try:
            epochs.append(int(float(parts[0])))
            box_loss.append(float(parts[1]))
            seg_loss.append(float(parts[2]))
            val_box_loss.append(float(parts[13]))
            val_seg_loss.append(float(parts[14]))
            map50_box.append(float(parts[7]))
            map50_mask.append(float(parts[11]))
        except:
            continue

    return epochs, box_loss, seg_loss, val_box_loss, val_seg_loss, map50_box, map50_mask


def plot_training(log_dir):
    """Plot the training curve"""
    log_file = Path(log_dir) / "results.csv"

    if not log_file.exists():
        print(f"""Not found: {log_file}""")
        return

    # Parse data
    epochs, box_loss, seg_loss, val_box_loss, val_seg_loss, map50_box, map50_mask = parse_results(
        log_file
    )

    if not epochs:
        print("""No data""")
        return

    # Create Chart
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # 1. Box Loss
    axes[0, 0].plot(epochs, box_loss, "b-", label="train")
    axes[0, 0].plot(epochs, val_box_loss, "r-", label="val")
    axes[0, 0].set_title("Box Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Seg Loss
    axes[0, 1].plot(epochs, seg_loss, "b-", label="train")
    axes[0, 1].plot(epochs, val_seg_loss, "r-", label="val")
    axes[0, 1].set_title("Seg Loss")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. mAP@0.5
    axes[1, 0].plot(epochs, map50_box, "g-", label="box", marker="o")
    axes[1, 0].plot(epochs, map50_mask, "m-", label="mask", marker="s")
    axes[1, 0].set_title("mAP@0.5")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0, 1])

    # 4. Overfitting assessment
    axes[1, 1].axis("off")

    # Calculation metrics
    best_idx = np.argmax(map50_mask) if map50_mask else 0
    best_epoch = epochs[best_idx] if best_idx < len(epochs) else 0
    best_map = map50_mask[best_idx] if best_idx < len(map50_mask) else 0

    # Determining overfitting
    overfit = False
    warning = ""

    if len(epochs) > 5:
        # Is the validation loss increasing?
        if val_box_loss[-1] > val_box_loss[-5]:
            overfit = True
            warning = """Validation loss is increasing, possible overfitting!"""

        # Hasn't the mAP improved in a long time?
        if len(epochs) - best_epoch > 10:
            overfit = True
            warning += f"""\n {len(epochs) - best_epoch} epochs without improving the best mAP"""

    # Show text
    info = f"epochs: {len(epochs)}\n"
    info += f"best_epoch: {best_epoch} (mAP={best_map:.4f})\n"
    info += f"current mAP: {map50_mask[-1]:.4f}\n\n"

    if overfit:
        info += warning
        axes[1, 1].set_facecolor("yellow")
    else:
        info += """Training is normal"""
        axes[1, 1].set_facecolor("lightgreen")

    axes[1, 1].text(0.1, 0.5, info, fontsize=12, verticalalignment="center")

    plt.tight_layout()

    # Save
    save_path = Path(log_dir) / "training_curve.png"
    plt.savefig(save_path, dpi=150)
    print(f"""Saved: {save_path}""")

    plt.show()


# Run
if __name__ == "__main__":
    plot_training(LOG_DIR)
