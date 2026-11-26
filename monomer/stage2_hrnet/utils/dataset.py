"""- ()
DNA"""

import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import cv2
from pathlib import Path
import random


class DNASegmentationDataset(Dataset):
    """DNA -"""

    def __init__(
        self,
        image_dir,
        anno_dir=None,
        heatmap_dir=None,
        input_size=(256, 256),
        num_keypoints=4,
        keypoint_names=None,
        transform=None,
        mode="train",
        augmentation=None,
    ):
        self.image_dir = Path(image_dir)
        self.anno_dir = Path(anno_dir) if anno_dir else None
        self.heatmap_dir = Path(heatmap_dir) if heatmap_dir else None
        self.input_size = input_size
        self.num_keypoints = num_keypoints
        self.mode = mode
        self.augmentation = augmentation or {}
        self.keypoint_names = keypoint_names or [f"dot_{i + 1}" for i in range(num_keypoints)]
        self.keypoint_order = {name: idx for idx, name in enumerate(self.keypoint_names)}

        self.image_files = sorted(
            [
                f
                for f in self.image_dir.iterdir()
                if f.suffix.lower() in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]
            ]
        )
        print(f"""[ {mode} ] {len(self.image_files)}""")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = self.load_image(img_path)
        keypoints = self.load_keypoints(img_path)

        orig_h, orig_w = image.shape
        target_h, target_w = self.input_size
        image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        scale_x = target_w / orig_w
        scale_y = target_h / orig_h
        keypoints[:, 0] *= scale_x
        keypoints[:, 1] *= scale_y

        if self.mode == "train":
            image, keypoints = self.apply_augmentation(image, keypoints)

        image = (image / 127.5) - 1.0  # [-1, 1]
        image = torch.from_numpy(image[np.newaxis, ...]).float()

        sample = {
            "image": image,
            "keypoints": keypoints,
            "image_path": str(img_path),
            "image_name": img_path.stem,
        }

        if self.heatmap_dir:
            hmp = self.heatmap_dir / f"{img_path.stem}_heatmap.npy"
            if hmp.exists():
                sample["heatmap"] = torch.from_numpy(np.load(hmp)).float()

        return sample

    def apply_augmentation(self, image, keypoints):
        h, w = image.shape

        if self.mode == "train":
            # ()
            if random.random() < self.augmentation.get("flip_lr", 0.5):
                image = cv2.flip(image, 1)
                keypoints[:, 0] = w - keypoints[:, 0]

            #
            if random.random() < self.augmentation.get("flip_ud", 0.1):
                image = cv2.flip(image, 0)
                keypoints[:, 1] = h - keypoints[:, 1]

            #
            rotation = self.augmentation.get("rotation", 15)
            if rotation > 0:
                angle = random.uniform(-rotation, rotation)
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                image = cv2.warpAffine(
                    image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0
                )
                visible = keypoints[:, 2] > 0
                if np.any(visible):
                    coords = keypoints[visible, :2]
                    coords_homo = np.column_stack([coords, np.ones(len(coords))])
                    keypoints[visible, :2] = coords_homo @ M.T

            #
            scale_range = self.augmentation.get("scale", [0.9, 1.1])
            if isinstance(scale_range, list) and len(scale_range) == 2:
                scale = random.uniform(scale_range[0], scale_range[1])
                new_w, new_h = int(w * scale), int(h * scale)
                scaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                image = np.zeros((h, w), dtype=np.float32)
                y_offset = (h - new_h) // 2
                x_offset = (w - new_w) // 2
                if scale > 1:
                    y_start = (new_h - h) // 2
                    x_start = (new_w - w) // 2
                    image = scaled[y_start : y_start + h, x_start : x_start + w]
                else:
                    image[
                        max(0, y_offset) : min(h, y_offset + new_h),
                        max(0, x_offset) : min(w, x_offset + new_w),
                    ] = scaled
                keypoints[:, 0] = keypoints[:, 0] * scale + x_offset
                keypoints[:, 1] = keypoints[:, 1] * scale + y_offset

        #
        if self.mode == "train":
            brightness = self.augmentation.get("brightness", 0.2)
            if brightness > 0 and random.random() < 0.5:
                beta = random.uniform(-brightness * 255, brightness * 255)
                image = np.clip(image + beta, 0, 255).astype(np.float32)

            contrast = self.augmentation.get("contrast", 0.2)
            if contrast > 0 and random.random() < 0.5:
                alpha = random.uniform(1 - contrast, 1 + contrast)
                image = np.clip(image * alpha, 0, 255).astype(np.float32)

            noise_scale = self.augmentation.get("gaussian_noise", 0.03)
            if noise_scale > 0 and random.random() < 0.3:
                noise = np.random.normal(0, noise_scale * 255, image.shape)
                image = np.clip(image + noise, 0, 255).astype(np.float32)

            if random.random() < self.augmentation.get("blur_prob", 0.5):
                ksize = self.augmentation.get("blur_kernel", 5) | 1
                sigma = random.uniform(0.5, 2.0)
                image = cv2.GaussianBlur(image, (ksize, ksize), sigma)

        return image, keypoints

    def load_image(self, img_path):
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f""": {img_path}""")
        return img.astype(np.float32)

    def load_keypoints(self, img_path):
        keypoints = np.zeros((self.num_keypoints, 3), dtype=np.float32)
        if self.anno_dir is None:
            return keypoints

        json_path = self.anno_dir / f"{img_path.stem}.json"
        if not json_path.exists():
            return keypoints

        try:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                return keypoints
            actual_height, actual_width = img.shape

            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            anno_width = data.get("imageWidth", actual_width)
            anno_height = data.get("imageHeight", actual_height)
            scale_x = actual_width / anno_width
            scale_y = actual_height / anno_height

            keypoint_order = self.keypoint_order

            for shape in data.get("shapes", []):
                label = shape.get("label", "").lower()
                if label in keypoint_order:
                    kp_idx = keypoint_order[label]
                    points = shape.get("points", [])
                    if len(points) > 0:
                        x, y = points[0]
                        if x == -1 and y == -1:
                            keypoints[kp_idx, 0] = 0
                            keypoints[kp_idx, 1] = 0
                            keypoints[kp_idx, 2] = 0
                        else:
                            keypoints[kp_idx, 0] = x * scale_x
                            keypoints[kp_idx, 1] = y * scale_y
                            keypoints[kp_idx, 2] = 2
        except Exception as e:
            print(f"""{json_path} : {e}""")

        return keypoints

    def get_num_dots(self, idx):
        keypoints = self.load_keypoints(self.image_files[idx])
        return int(np.sum(keypoints[:, 2] > 0))


def get_data_loaders(config):
    """"""
    data_cfg = config["data"]
    train_subdir = data_cfg.get("train_subdir", "train")
    val_subdir = data_cfg.get("val_subdir", "val")

    anno_subdir = (
        train_subdir
        if os.path.exists(os.path.join(data_cfg["anno_dir"], train_subdir))
        else train_subdir.replace("_aug", "")
    )

    aug_cfg = config.get("augmentation", {})

    val_anno_subdir = (
        val_subdir
        if os.path.exists(os.path.join(data_cfg["anno_dir"], val_subdir))
        else val_subdir.replace("_aug", "")
    )

    train_dataset = DNASegmentationDataset(
        image_dir=os.path.join(data_cfg["processed_dir"], train_subdir),
        anno_dir=os.path.join(data_cfg["anno_dir"], anno_subdir),
        heatmap_dir=os.path.join(data_cfg["heatmap_dir"], train_subdir)
        if data_cfg.get("heatmap_dir")
        and os.path.exists(os.path.join(data_cfg["heatmap_dir"], train_subdir))
        else None,
        keypoint_names=data_cfg.get("keypoint_names"),
        input_size=tuple(data_cfg["input_size"]),
        num_keypoints=data_cfg["num_keypoints"],
        mode="train",
        augmentation=aug_cfg,
    )

    val_dataset = DNASegmentationDataset(
        image_dir=os.path.join(data_cfg["processed_dir"], val_subdir),
        anno_dir=os.path.join(data_cfg["anno_dir"], val_anno_subdir),
        heatmap_dir=os.path.join(data_cfg["heatmap_dir"], val_subdir)
        if data_cfg.get("heatmap_dir")
        and os.path.exists(os.path.join(data_cfg["heatmap_dir"], val_subdir))
        else None,
        keypoint_names=data_cfg.get("keypoint_names"),
        input_size=tuple(data_cfg["input_size"]),
        num_keypoints=data_cfg["num_keypoints"],
        mode="val",
        augmentation=aug_cfg,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
        pin_memory=True,
    )

    return train_loader, val_loader


def collate_fn(batch):
    images = torch.stack([item["image"] for item in batch])
    keypoints = [item["keypoints"] for item in batch]
    image_paths = [item["image_path"] for item in batch]
    image_names = [item["image_name"] for item in batch]
    result = {
        "image": images,
        "keypoints": keypoints,
        "image_path": image_paths,
        "image_name": image_names,
    }
    if "heatmap" in batch[0]:
        result["heatmap"] = torch.stack([item["heatmap"] for item in batch])
    return result
