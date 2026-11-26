"""HRNet Training Script
Used to train the DNA white spot detection model"""

import os
import sys
import yaml
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from pathlib import Path
from datetime import datetime

# Add project path
sys.path.insert(0, str(Path(__file__).parent))


class Logger:
    """Custom log class, output to both console and file"""

    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log_file = log_file
        self.log = open(log_file, "w", encoding="utf-8")
        print(f"""Log file: {log_file}""")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


from models import HRNet, SimpleHRNet, hrnet_w32_cfg
from utils import DNASegmentationDataset, get_data_loaders
from utils import generate_heatmap, heatmap_to_coordinates
from utils import MetricsTracker, compute_pck, compute_oks
from utils import plot_training_curves, visualize_predictions


def set_seed(seed):
    """Set a random seed to ensure reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path):
    """Load configuration file"""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def get_model(config):
    """Get the model based on the configuration"""
    model_type = config["model"].get("hrnet_type", "SimpleHRNet")
    num_keypoints = config["data"]["num_keypoints"]

    if model_type == "SimpleHRNet":
        model = SimpleHRNet(num_keypoints=num_keypoints, in_channels=1)
        print(f"""Using the simplified version of HRNet, number of key points: {num_keypoints}""")
    elif model_type == "HRNet-W32":
        cfg = hrnet_w32_cfg.copy()
        cfg["NUM_HEATMAPS"] = num_keypoints
        model = HRNet(cfg, output_size=tuple(config["heatmap"]["output_size"]))
        print(f"""Using HRNet-W32, number of keypoints: {num_keypoints}""")

        # Load pre-trained weights
        pretrained_path = config["model"].get("pretrained")
        if pretrained_path and Path(pretrained_path).exists():
            print(f"""Loading pre-trained weights: {pretrained_path}""")
            try:
                pretrained_dict = torch.load(pretrained_path, map_location="cpu")

                # Handling pre-trained weights in different formats
                if "state_dict" in pretrained_dict:
                    pretrained_dict = pretrained_dict["state_dict"]

                model_dict = model.state_dict()

                # Filter out mismatched layers (the final classification layer)
                pretrained_filtered = {}
                for k, v in pretrained_dict.items():
                    # Remove the 'module.' prefix (if present)
                    k = k.replace("module.", "")
                    if k in model_dict and model_dict[k].shape == v.shape:
                        pretrained_filtered[k] = v
                    else:
                        print(f"""Skip: {k}""")

                # Load matching weights
                model_dict.update(pretrained_filtered)
                model.load_state_dict(model_dict, strict=False)
                print(
                    f"""Successfully loaded {len(pretrained_filtered)} / {len(model_dict)} layer weights"""
                )
            except Exception as e:
                print(f"""Failed to load pre-trained weights: {e}""")
                print("""Training will start from scratch.""")
        else:
            print("""No pre-trained weights found. Training will start from scratch.""")
    else:
        raise ValueError(f"""Unknown model type: {model_type}""")

    return model


def get_optimizer(model, config):
    """Get optimizer"""
    opt_cfg = config["training"]["optimizer"]
    optimizer_type = opt_cfg["type"].lower()
    lr = opt_cfg["lr"]
    weight_decay = opt_cfg.get("weight_decay", 0)

    if optimizer_type == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optimizer_type == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"""Unknown optimizer type: {optimizer_type}""")

    return optimizer


def get_scheduler(optimizer, config):
    """Get learning rate scheduler"""
    sched_cfg = config["training"]["scheduler"]
    scheduler_type = sched_cfg["type"].lower()

    if scheduler_type == "steplr":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=sched_cfg.get("step_size", 30), gamma=sched_cfg.get("factor", 0.1)
        )
    elif scheduler_type == "cosineannealinglr":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=sched_cfg.get("T_max", config["training"]["epochs"]),
            eta_min=sched_cfg.get("eta_min", 0),
        )
    elif scheduler_type == "reducelronplateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=sched_cfg.get("patience", 10),
            factor=sched_cfg.get("factor", 0.5),
            min_lr=sched_cfg.get("min_lr", 1e-7),
        )
    else:
        scheduler = None

    return scheduler


def train_epoch(model, train_loader, criterion, optimizer, device, config, epoch, scaler=None):
    """Train one epoch (supports mixed-precision training)"""
    model.train()
    total_loss = 0
    num_batches = len(train_loader)
    use_amp = config["training"].get("use_amp", False) and scaler is not None

    # Heatmap parameters
    heatmap_size = tuple(config["heatmap"]["output_size"])
    sigma = config["heatmap"]["sigma"]
    num_keypoints = config["data"]["num_keypoints"]

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    for batch_idx, batch in enumerate(pbar):
        images = batch["image"].to(device)
        keypoints = batch["keypoints"]

        # Generate heat map GT and visibility mask
        batch_size = images.size(0)
        heatmaps_gt = []
        visibility_masks = []
        for i in range(batch_size):
            kp = keypoints[i].cpu().numpy()
            # Scale keypoint coordinates to heat map size
            kp_scaled = kp.copy()
            kp_scaled[:, 0] = kp[:, 0] * (heatmap_size[1] / config["data"]["input_size"][1])
            kp_scaled[:, 1] = kp[:, 1] * (heatmap_size[0] / config["data"]["input_size"][0])

            hm = generate_heatmap(kp_scaled, heatmap_size, sigma, num_keypoints)
            heatmaps_gt.append(hm)

            # Create a visibility mask (visibility=2 indicates visible, mask=1)
            vis_mask = (kp[:, 2] > 0).astype(np.float32)  # [num_keypoints]
            visibility_masks.append(vis_mask)

        heatmaps_gt = torch.from_numpy(np.stack(heatmaps_gt)).float().to(device)
        visibility_masks = (
            torch.from_numpy(np.stack(visibility_masks)).float().to(device)
        )  # [B, num_keypoints]

        # Forward propagation (mixed precision)
        optimizer.zero_grad()

        with autocast(enabled=use_amp):
            heatmaps_pred = model(images)

            # Calculate loss (using visibility mask and negative sample supervision)
            # Expand the mask to the size of the heatmap [B, num_keypoints, H, W]
            mask_expanded = visibility_masks.unsqueeze(-1).unsqueeze(-1).expand_as(heatmaps_gt)

            # 1. Positive sample loss: Keypoints that exist; MSE is calculated normally.
            diff = (heatmaps_pred - heatmaps_gt) ** 2
            pos_loss = (diff * mask_expanded).sum() / (mask_expanded.sum() + 1e-8)

            # 2. Negative sample loss: Keypoints that do not exist; penalize non-zero outputs
            # Create a negative sample mask (1 - visibility)
            neg_mask = (1 - visibility_masks).unsqueeze(-1).unsqueeze(-1).expand_as(heatmaps_gt)
            # For non-existent points, the GT is 0, and the penalty is applied when the predicted value is far from 0.
            neg_loss = ((heatmaps_pred**2) * neg_mask).sum() / (neg_mask.sum() + 1e-8)

            # 3. Total loss (configurable weights)
            neg_weight = config["training"]["loss"].get("neg_weight", 0.5)
            loss = pos_loss + neg_weight * neg_loss

        # Backpropagation (mixed precision)
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

        # Update progress bar
        pbar.set_postfix(
            {"loss": f"{loss.item():.4f}", "avg_loss": f"{total_loss / (batch_idx + 1):.4f}"}
        )

    avg_loss = total_loss / num_batches
    return avg_loss


def validate(model, val_loader, criterion, device, config, scaler=None):
    """Validation (supports mixed precision)"""
    model.eval()
    total_loss = 0
    all_pred_keypoints = []
    all_gt_keypoints = []

    heatmap_size = tuple(config["heatmap"]["output_size"])
    sigma = config["heatmap"]["sigma"]
    num_keypoints = config["data"]["num_keypoints"]
    input_size = tuple(config["data"]["input_size"])
    threshold = config["inference"]["heatmap_threshold"]
    use_amp = config["training"].get("use_amp", False)

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="[Validate]")
        for batch in pbar:
            images = batch["image"].to(device)
            keypoints_gt = batch["keypoints"]

            batch_size = images.size(0)

            # Generate heat map GT and visibility mask
            heatmaps_gt = []
            visibility_masks = []
            for i in range(batch_size):
                kp = keypoints_gt[i].cpu().numpy()
                kp_scaled = kp.copy()
                kp_scaled[:, 0] = kp[:, 0] * (heatmap_size[1] / input_size[1])
                kp_scaled[:, 1] = kp[:, 1] * (heatmap_size[0] / input_size[0])
                hm = generate_heatmap(kp_scaled, heatmap_size, sigma, num_keypoints)
                heatmaps_gt.append(hm)

                # Create a visibility mask
                vis_mask = (kp[:, 2] > 0).astype(np.float32)
                visibility_masks.append(vis_mask)

            heatmaps_gt = torch.from_numpy(np.stack(heatmaps_gt)).float().to(device)
            visibility_masks = torch.from_numpy(np.stack(visibility_masks)).float().to(device)

            # Forward propagation (mixed precision)
            with autocast(enabled=use_amp):
                heatmaps_pred = model(images)

                # Calculate loss (using visibility mask and negative sample supervision)
                mask_expanded = visibility_masks.unsqueeze(-1).unsqueeze(-1).expand_as(heatmaps_gt)

                # 1. Positive sample loss
                diff = (heatmaps_pred - heatmaps_gt) ** 2
                pos_loss = (diff * mask_expanded).sum() / (mask_expanded.sum() + 1e-8)

                # 2. Negative sample loss
                neg_mask = (1 - visibility_masks).unsqueeze(-1).unsqueeze(-1).expand_as(heatmaps_gt)
                neg_loss = ((heatmaps_pred**2) * neg_mask).sum() / (neg_mask.sum() + 1e-8)

                # 3. Total loss
                neg_weight = config["training"]["loss"].get("neg_weight", 0.5)
                loss = pos_loss + neg_weight * neg_loss
            total_loss += loss.item()

            # Extract keypoint coordinates from the heatmap
            for i in range(batch_size):
                pred_coords = heatmap_to_coordinates(
                    heatmaps_pred[i].cpu().numpy(),
                    threshold=threshold,
                    cross_channel_nms=(num_keypoints != 6),
                )

                # Convert back to the input image coordinate system
                pred_kp = np.zeros((num_keypoints, 3))
                for j, (x, y, conf) in enumerate(pred_coords):
                    if conf > 0:
                        pred_kp[j, 0] = x * (input_size[1] / heatmap_size[1])
                        pred_kp[j, 1] = y * (input_size[0] / heatmap_size[0])
                        pred_kp[j, 2] = 2 if conf > threshold else 0

                all_pred_keypoints.append(pred_kp)
                all_gt_keypoints.append(keypoints_gt[i].cpu().numpy())

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / len(val_loader)

    # Calculate evaluation metrics
    pck = compute_pck(all_pred_keypoints, all_gt_keypoints, input_size, threshold=0.05)
    oks = compute_oks(all_pred_keypoints, all_gt_keypoints, input_size)

    # Calculate the classification accuracy for the number of white spots
    pred_counts = [int(np.sum(kp[:, 2] > 0)) for kp in all_pred_keypoints]
    gt_counts = [int(np.sum(kp[:, 2] > 0)) for kp in all_gt_keypoints]
    acc = np.mean(np.array(pred_counts) == np.array(gt_counts))

    metrics = {"loss": avg_loss, "pck": pck, "oks": oks, "count_acc": acc}

    return metrics, all_pred_keypoints, all_gt_keypoints


def main():
    parser = argparse.ArgumentParser(description="Train HRNet for DNA dot detection")
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml", help="path to config file"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="path to checkpoint to resume from"
    )
    parser.add_argument("--device", type=str, default="cuda", help="device to use (cuda or cpu)")
    args = parser.parse_args()

    # Create a run directory with a timestamp (YOLO-style runs/exp)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"train_{timestamp}"
    runs_dir = Path("runs")
    run_dir = runs_dir / run_name
    weights_dir = run_dir / "weights"
    logs_dir = run_dir / "logs"
    viz_dir = run_dir / "visualizations"

    for d in [runs_dir, run_dir, weights_dir, logs_dir, viz_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Create log file (in the run directory)
    log_file = logs_dir / "train.log"
    sys.stdout = Logger(str(log_file))

    print("=" * 60)
    print(f"""Training run: {run_name}""")
    print(f"""Save directory: {run_dir}""")
    print("=" * 60)

    # Load Configuration
    config = load_config(args.config)
    print("""\n Configuration loaded successfully:""")
    print(yaml.dump(config, default_flow_style=False))

    # Back up configuration to the runtime directory
    import shutil

    shutil.copy(args.config, run_dir / "config.yaml")

    # Set random seed
    set_seed(config["project"]["seed"])

    # Update the output directory to the working directory
    output_dir = weights_dir
    log_dir = logs_dir

    # Set up device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"""Device in use: {device}""")

    # Create a data loader
    print("""\n Loading data...""")
    train_loader, val_loader = get_data_loaders(config)

    # Create model
    print("""\n Creating model...""")
    model = get_model(config).to(device)

    # Print model parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"""Total number of parameters: {total_params:,}""")
    print(f"""Number of trainable parameters: {trainable_params:,}""")

    # Loss function
    criterion = nn.MSELoss()

    # Optimizer
    optimizer = get_optimizer(model, config)

    # Learning Rate Scheduler
    scheduler = get_scheduler(optimizer, config)

    # TensorBoard
    writer = SummaryWriter(log_dir)

    # Metric Tracking
    metrics_tracker = MetricsTracker()

    # Initialize mixed precision training scaler
    use_amp = config["training"].get("use_amp", False)
    scaler = GradScaler() if use_amp and device.type == "cuda" else None
    if use_amp:
        print(
            "Mixed precision training: "
            + ("enabled (AMP)" if scaler is not None else "disabled (CUDA required)")
        )

    # Training Parameters
    epochs = config["training"]["epochs"]
    save_interval = config["output"]["save_interval"]
    val_interval = config["training"]["val_interval"]
    best_val_pck = 0.0
    best_val_loss = float("inf")
    patience_counter = 0
    early_stop_patience = config["training"]["early_stopping"]["patience"]

    # Training History
    history = {"train_loss": [], "val_loss": [], "val_pck": [], "val_oks": [], "val_count_acc": []}

    print("""\n Start training...""")
    for epoch in range(1, epochs + 1):
        # Training
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, config, epoch, scaler
        )
        history["train_loss"].append(train_loss)

        print(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.4f}")
        writer.add_scalar("Loss/train", train_loss, epoch)
        metrics_tracker.update(train_loss=train_loss)

        # Validation
        if epoch % val_interval == 0:
            val_metrics, pred_kps, gt_kps = validate(
                model, val_loader, criterion, device, config, scaler
            )
            val_loss = val_metrics["loss"]

            print(
                f"Validation - Loss: {val_loss:.4f}, "
                f"PCK: {val_metrics['pck']:.4f}, "
                f"OKS: {val_metrics['oks']:.4f}, "
                f"Count Acc: {val_metrics['count_acc']:.4f}"
            )

            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("Metrics/PCK", val_metrics["pck"], epoch)
            writer.add_scalar("Metrics/OKS", val_metrics["oks"], epoch)
            writer.add_scalar("Metrics/Count_Acc", val_metrics["count_acc"], epoch)

            history["val_loss"].append(val_loss)
            history["val_pck"].append(val_metrics["pck"])
            history["val_oks"].append(val_metrics["oks"])
            history["val_count_acc"].append(val_metrics["count_acc"])

            metrics_tracker.set_epoch_metrics(epoch, **val_metrics)

            # Save the best model
            pck = val_metrics["pck"]
            if pck > best_val_pck:
                best_val_pck = pck
                patience_counter = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": val_loss,
                        "val_pck": pck,
                        "config": config,
                    },
                    output_dir / "best_model.pth",
                )
                print(f"""[OK] Save the best model (PCK= {pck:.4f}, val_loss= {val_loss:.4f})""")
            else:
                patience_counter += 1

            # Learning rate scheduling
            if scheduler is not None:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

            # Visualize some validation results
            if epoch % (val_interval * 2) == 0:
                # Visualize the first 4 samples
                val_dataset = val_loader.dataset
                vis_images = []
                vis_pred_kps = []
                vis_gt_kps = []
                for i in range(min(4, len(val_dataset))):
                    sample = val_dataset[i]
                    vis_images.append(sample["image"])
                    vis_gt_kps.append(sample["keypoints"])
                    vis_pred_kps.append(pred_kps[i])

                viz_path = viz_dir / f"val_epoch_{epoch}.png"
                # The process is simplified here, but in reality, the batch data should be reorganized.

        # Save checkpoints periodically
        if epoch % save_interval == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "config": config,
                },
                output_dir / f"checkpoint_epoch_{epoch}.pth",
            )

        # Early stopping check
        if patience_counter >= early_stop_patience:
            print(f"""\n Early stopping! No improvement for {early_stop_patience} epochs.""")
            break

    # Save the final model
    torch.save(
        {
            "epoch": epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
        },
        output_dir / "final_model.pth",
    )

    # Plot the training curve
    plot_path = viz_dir / "training_curves.png"
    metrics_dict = {
        "PCK": history["val_pck"],
        "OKS": history["val_oks"],
        "Count Acc": history["val_count_acc"],
    }
    plot_training_curves(
        history["train_loss"], history["val_loss"], metrics_dict, save_path=plot_path
    )

    # Save Training Results Summary
    results = {
        "run_name": run_name,
        "best_val_pck": float(best_val_pck),
        "final_train_loss": float(history["train_loss"][-1]),
        "epochs_trained": len(history["train_loss"]),
        "config": config,
        "history": {k: [float(v) for v in vals] for k, vals in history.items()},
    }

    # Save JSON results
    import json

    with open(run_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Save CSV training log
    import csv

    csv_path = run_dir / "training_log.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Train Loss", "Val Loss", "PCK", "OKS", "Count Acc"])
        for i in range(len(history["train_loss"])):
            row = [
                i + 1,
                history["train_loss"][i],
                history["val_loss"][i] if i < len(history["val_loss"]) else "",
                history["val_pck"][i] if i < len(history["val_pck"]) else "",
                history["val_oks"][i] if i < len(history["val_oks"]) else "",
                history["val_count_acc"][i] if i < len(history["val_count_acc"]) else "",
            ]
            writer.writerow(row)

    # Print summary
    print("\n" + "=" * 60)
    print("""Training completed!""")
    print(f"""Run Directory: {run_dir}""")
    print(f"""Best PCK: {best_val_pck:.4f}""")
    print(f"""Final training loss: {history["train_loss"][-1]:.4f}""")
    print(f"""Number of training rounds: {len(history["train_loss"])}""")
    print("=" * 60)
    print(f"""\n Result file:""")
    print(f"""- Optimal weight: {weights_dir / "best_model.pth"}""")
    print(f"""- Final weight: {weights_dir / "final_model.pth"}""")
    print(f"""- Training curve: {plot_path}""")
    print(f"""- Training log: {csv_path}""")
    print(f"""- Results Summary: {run_dir / "results.json"}""")
    print("=" * 60)
    writer.close()


if __name__ == "__main__":
    main()
