# -*- coding: utf-8 -*-
"""HRNet DNA monomer keypoint detection training script"""

import os, sys, time, shutil
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
import yaml
from tqdm import tqdm

script_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(script_dir))

from models import SimpleHRNet
from utils import DNASegmentationDataset, get_data_loaders
from utils import generate_heatmap, heatmap_to_coordinates
from utils import MetricsTracker, compute_pck, compute_oks
from utils import plot_training_curves


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


import random


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_model(config, device):
    num_kpts = config["data"]["num_keypoints"]
    from copy import deepcopy
    from models import HRNet, hrnet_w32_cfg

    model_cfg = deepcopy(hrnet_w32_cfg)
    model_cfg["NUM_HEATMAPS"] = num_kpts
    model = HRNet(model_cfg)
    pretrained_path = config["model"].get("pretrained")
    if pretrained_path and Path(pretrained_path).exists():
        state = torch.load(pretrained_path, map_location="cpu")
        if "state_dict" in state:
            state = state["state_dict"]
        elif "model" in state:
            state = state["model"]
        if "conv1.weight" in state and state["conv1.weight"].shape[1] == 3:
            state["conv1.weight"] = state["conv1.weight"].mean(dim=1, keepdim=True)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"Loaded: missing={len(missing)}, unexpected={len(unexpected)}")
    model = model.to(device)
    return model


def get_optimizer(model, config):
    oc = config["training"]["optimizer"]
    t = oc["type"].lower()
    if t == "adam":
        return torch.optim.Adam(
            model.parameters(), lr=oc["lr"], weight_decay=oc.get("weight_decay", 0)
        )
    return torch.optim.SGD(
        model.parameters(), lr=oc["lr"], weight_decay=oc.get("weight_decay", 0), momentum=0.9
    )


def get_scheduler(optimizer, config):
    sc = config["training"]["scheduler"]
    if sc["type"].lower() == "cosineannealinglr":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=sc.get("T_max", config["training"]["epochs"]),
            eta_min=sc.get("eta_min", 0),
        )
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)


class HeatmapLoss(nn.Module):
    def __init__(self, bg_weight=0.1):
        super().__init__()
        self.bg_weight = bg_weight

    def forward(self, pred, target):
        loss = nn.functional.mse_loss(pred, target, reduction="none")
        bg = (target < 0.01).float()
        fg = (target >= 0.01).float()
        return (loss * (fg + self.bg_weight * bg)).mean()


def generate_heatmap_gpu(keypoints, heatmap_size, sigma, num_keypoints):
    """GPU-based heatmap generation - much faster than CPU version"""
    h, w = heatmap_size
    kps = keypoints if torch.is_tensor(keypoints) else torch.from_numpy(keypoints).float()
    device = kps.device
    kps = kps.to(device)
    hm = torch.zeros((num_keypoints, h, w), device=device)

    # Create coordinate grids (once and cache)
    ys, xs = torch.meshgrid(
        torch.arange(h, device=device, dtype=torch.float32),
        torch.arange(w, device=device, dtype=torch.float32),
        indexing="ij",
    )

    for i in range(num_keypoints):
        x, y, v = kps[i]
        if v > 0 and x >= 0 and y >= 0:
            hm[i] = torch.exp(-((xs - x) ** 2 + (ys - y) ** 2) / (2 * sigma**2))

    return hm


def train_epoch(
    model, loader, criterion, optimizer, device, config, epoch, scaler=None, log_func=print
):
    model.train()
    amp_on = config["training"].get("use_amp", False) and scaler is not None
    hm_size = tuple(config["heatmap"]["output_size"])
    sigma = config["heatmap"]["sigma"]
    nk = config["data"]["num_keypoints"]
    total_loss = 0
    pbar = tqdm(loader, desc=f"Epoch {epoch}")

    # Pre-create meshgrid on GPU for heatmap generation
    h, w = hm_size
    hm_ys, hm_xs = torch.meshgrid(
        torch.arange(h, device=device, dtype=torch.float32),
        torch.arange(w, device=device, dtype=torch.float32),
        indexing="ij",
    )

    for batch in pbar:
        imgs = batch["image"].to(device)
        kps = batch["keypoints"].to(device)
        bs = imgs.size(0)

        # Scale keypoints to heatmap resolution
        kps_scaled = kps.clone()
        kps_scaled[:, :, 0] = kps[:, :, 0] * hm_size[1] / config["data"]["input_size"][1]
        kps_scaled[:, :, 1] = kps[:, :, 1] * hm_size[0] / config["data"]["input_size"][0]

        # Generate heatmaps on GPU
        hms = []
        for b in range(bs):
            hm = torch.zeros((nk, h, w), device=device)
            for i in range(nk):
                x, y, v = kps_scaled[b, i]
                if v > 0 and x >= 0 and y >= 0:
                    hm[i] = torch.exp(-((hm_xs - x) ** 2 + (hm_ys - y) ** 2) / (2 * sigma**2))
            hms.append(hm)
        hms = torch.stack(hms)

        if amp_on:
            with torch.cuda.amp.autocast():
                pred = model(imgs)
                loss = criterion(pred, hms)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(imgs)
            loss = criterion(pred, hms)
            loss.backward()
            optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
        pbar.set_postfix({"loss": loss.item()})

    return total_loss / len(loader)


def validate(model, loader, criterion, device, config, log_func=print):
    model.eval()
    hm_size = tuple(config["heatmap"]["output_size"])
    sigma = config["heatmap"]["sigma"]
    nk = config["data"]["num_keypoints"]
    th = config["inference"]["heatmap_threshold"]
    total_loss = 0
    ap, ag = [], []

    h, w = hm_size
    hm_ys, hm_xs = torch.meshgrid(
        torch.arange(h, device=device, dtype=torch.float32),
        torch.arange(w, device=device, dtype=torch.float32),
        indexing="ij",
    )

    with torch.no_grad():
        for batch in tqdm(loader, desc="Val"):
            imgs = batch["image"].to(device)
            kps = batch["keypoints"].to(device)
            bs = imgs.size(0)

            kps_scaled = kps.clone()
            kps_scaled[:, :, 0] = kps[:, :, 0] * hm_size[1] / config["data"]["input_size"][1]
            kps_scaled[:, :, 1] = kps[:, :, 1] * hm_size[0] / config["data"]["input_size"][0]

            hms = []
            for b in range(bs):
                hm = torch.zeros((nk, h, w), device=device)
                for i in range(nk):
                    x, y, v = kps_scaled[b, i]
                    if v > 0 and x >= 0 and y >= 0:
                        hm[i] = torch.exp(-((hm_xs - x) ** 2 + (hm_ys - y) ** 2) / (2 * sigma**2))
                hms.append(hm)
            hms = torch.stack(hms)

            pred_hms = model(imgs)
            total_loss += criterion(pred_hms, hms).item()

            for b in range(bs):
                pk = heatmap_to_coordinates(pred_hms[b].cpu().numpy(), th)
                gk = kps_scaled[b, :, :].cpu().numpy()
                ap.append(pk)
                ag.append(gk)

    ap = np.array(ap)
    ag = np.array(ag)
    return {
        "loss": total_loss / len(loader),
        "pck": compute_pck(ap, ag, (h, w), threshold=0.2),
        "oks": compute_oks(ap, ag, (h, w)),
    }


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--type", choices=["0", "00", "01", "1", "10", "11"])
    parser.add_argument("--config")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.config is None:
        if args.type is None:
            parser.error("provide --type or --config")
        args.config = str(
            Path(__file__).resolve().parent / "configs" / "types" / f"type_{args.type}.yaml"
        )

    config = load_config(args.config)
    print(yaml.dump(config, default_flow_style=False))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    set_seed(config["project"]["seed"])

    run_name = f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir = Path(config["output"]["weights_dir"]) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(config["output"]["logs_dir"]).mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, out_dir / "config.yaml")

    # Set up log file
    log_file = Path(config["output"]["logs_dir"]) / "train_log.txt"
    log_fp = open(log_file, "a", encoding="utf-8")

    def log(msg):
        timestamp = time.strftime("%H:%M:%S")
        full = f"[{timestamp}] {msg}"
        print(full)
        log_fp.write(full + "\n")
        log_fp.flush()

    log(f"Starting training run: {run_name}")
    log(f"Device: {device}")

    log("Loading data...")
    tl, vl = get_data_loaders(config)
    log(f"Train samples: {len(tl.dataset)}, Val samples: {len(vl.dataset)}")

    model = get_model(config, device)
    log(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    criterion = HeatmapLoss(bg_weight=config["heatmap"]["bg_weight"])
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)

    amp_on = config["training"].get("use_amp", False)
    scaler = GradScaler() if amp_on and device.type == "cuda" else None
    if amp_on:
        log("AMP enabled")

    epochs = config["training"]["epochs"]
    si = config["output"]["save_interval"]
    vi = config["training"]["val_interval"]
    esp = config["training"]["early_stopping"]["patience"]
    best_pck = 0.0
    best_loss = float("inf")
    pc = 0
    hist = {"train_loss": [], "val_loss": [], "pck": [], "oks": []}

    log(f"Training {epochs} epochs...")
    train_start = time.time()

    for ep in range(1, epochs + 1):
        ep_start = time.time()

        tl_loss = train_epoch(model, tl, criterion, optimizer, device, config, ep, scaler, log)
        hist["train_loss"].append(tl_loss)
        lr = optimizer.param_groups[0]["lr"]
        ep_time = time.time() - ep_start

        if ep % vi == 0 or ep == epochs:
            vm = validate(model, vl, criterion, device, config, log)
            hist["val_loss"].append(vm["loss"])
            hist["pck"].append(vm["pck"])
            hist["oks"].append(vm["oks"])

            msg = (
                f"Ep {ep}/{epochs} Train:{tl_loss:.4f} Val:{vm['loss']:.4f} "
                f"PCK:{vm['pck']:.4f} OKS:{vm['oks']:.4f} LR:{lr:.6f} "
                f"Time:{ep_time:.0f}s"
            )
            log(msg)

            if not isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
                scheduler.step(vm["loss"])

            if vm["pck"] > best_pck:
                best_pck = vm["pck"]
                pc = 0
                torch.save(
                    {
                        "epoch": ep,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "config": config,
                    },
                    out_dir / "best_model.pth",
                )
                log(f"Saved best model PCK={best_pck:.4f}")
            else:
                pc += 1

            if vm["loss"] < best_loss:
                best_loss = vm["loss"]
        else:
            log(f"Ep {ep}/{epochs} Loss:{tl_loss:.4f} LR:{lr:.6f} Time:{ep_time:.0f}s")

        if isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
            scheduler.step()

        if ep % si == 0:
            ckpt_path = out_dir / f"checkpoint_epoch_{ep}.pth"
            torch.save({"epoch": ep, "model": model.state_dict()}, ckpt_path)
            log(f"Saved checkpoint: {ckpt_path}")

        if pc >= esp:
            log(f"Early stopping at epoch {ep}")
            break

    total_time = time.time() - train_start
    final_ckpt = out_dir / "final_model.pth"
    torch.save(
        {
            "epoch": epochs,
            "model": model.state_dict(),
            "best_pck": best_pck,
            "config": config,
            "history": {k: [float(v) for v in vals] for k, vals in hist.items()},
        },
        final_ckpt,
    )

    plot_training_curves(hist, str(out_dir / "training_curves.png"))

    log(f"Training complete! Best PCK: {best_pck:.4f}")
    log(f"Total time: {total_time:.0f}s ({total_time / 60:.1f}min)")
    log(f"Results saved to: {out_dir}")

    log_fp.close()


if __name__ == "__main__":
    main()
