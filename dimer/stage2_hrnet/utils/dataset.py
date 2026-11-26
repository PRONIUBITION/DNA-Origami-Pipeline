"""Dataset Class Definition - OpenCV Optimized Version
Used to load DNA origami images and white dot annotations"""

import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
from pathlib import Path
import random


class DNASegmentationDataset(Dataset):
    """DNA White Dot Detection Dataset (OpenCV Optimized Version)
    Supports loading from Labelme annotation files"""

    def __init__(
        self,
        image_dir,
        anno_dir=None,
        heatmap_dir=None,
        input_size=(256, 256),
        num_keypoints=4,
        transform=None,
        mode="train",
        augmentation=None,
    ):
        """Args:
        image_dir: Path to the image folder
        anno_dir: Annotation folder path (Labelme JSON format)
        heatmap_dir: Path to the pre-generated heatmap folder (optional)
        input_size: Input image size (H, W)
        num_keypoints: Number of keypoints (maximum number of small white dots)
        transform: Data augmentation transformation (deprecated, retained for compatibility)
        mode: 'train', 'val', or 'test'
        augmentation: Augmentation configuration dictionary"""
        self.image_dir = Path(image_dir)
        self.anno_dir = Path(anno_dir) if anno_dir else None
        self.heatmap_dir = Path(heatmap_dir) if heatmap_dir else None
        self.input_size = input_size
        self.num_keypoints = num_keypoints
        self.mode = mode
        self.augmentation = augmentation or {}

        # Get all image files
        self.image_files = sorted(
            [
                f
                for f in self.image_dir.iterdir()
                if f.suffix.lower() in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]
            ]
        )

        print(f"""[ {mode} ] Found {len(self.image_files)} images""")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load Image
        img_path = self.image_files[idx]
        image = self.load_image(img_path)

        # Load annotations (relative to the original image size)
        keypoints = self.load_keypoints(img_path)

        # Resize image and keypoints to input_size
        orig_h, orig_w = image.shape
        target_h, target_w = self.input_size

        # Resize images using OpenCV
        image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        # Synchronously scale keypoint coordinates
        scale_x = target_w / orig_w
        scale_y = target_h / orig_h
        keypoints[:, 0] *= scale_x
        keypoints[:, 1] *= scale_y

        # Data augmentation (used for both training and validation sets)
        if self.mode == "train":
            image, keypoints = self.apply_augmentation(image, keypoints)

        # Normalization to [-1, 1]
        image = (image / 127.5) - 1.0

        # Convert to Tensor [C, H, W]
        image = torch.from_numpy(image[np.newaxis, ...]).float()

        sample = {
            "image": image,
            "keypoints": keypoints,
            "image_path": str(img_path),
            "image_name": img_path.stem,
        }

        # If a pre-generated heatmap exists, load it
        if self.heatmap_dir:
            heatmap_path = self.heatmap_dir / f"{img_path.stem}_heatmap.npy"
            if heatmap_path.exists():
                heatmap = np.load(heatmap_path)
                sample["heatmap"] = torch.from_numpy(heatmap).float()

        return sample

    def apply_augmentation(self, image, keypoints):
        """Apply OpenCV data augmentation to simultaneously transform images and keypoints
        2-3 times faster than PIL"""
        h, w = image.shape

        # ===== Geometric Transformation (Keypoint Synchronization) =====
        if self.mode == "train":
            # Random horizontal flip
            if random.random() < self.augmentation.get("flip_lr", 0.5):
                image = cv2.flip(image, 1)  # 1 = Horizontal flip
                # Synchronously flip the x-coordinate of the keypoint
                keypoints[:, 0] = w - keypoints[:, 0]
                # Swap left and right keypoints
                keypoints[[0, 2]] = keypoints[[2, 0]]  # left_dot_1 <-> right_dot_1
                keypoints[[1, 3]] = keypoints[[3, 1]]  # left_dot_2 <-> right_dot_2

            # Random vertical flip (low symmetry of DNA structure between top and bottom, probability set low)
            if random.random() < self.augmentation.get("flip_ud", 0.1):
                image = cv2.flip(image, 0)  # 0 = Vertical flip
                # Simultaneously flip the y-coordinate of the keypoint
                keypoints[:, 1] = h - keypoints[:, 1]

            # Random rotation (15 degrees)
            rotation = self.augmentation.get("rotation", 15)
            rotation = self.augmentation.get("rotation", 30)
            # rotation = self.augmentation.get('rotation', 20)
            if rotation > 0:
                angle = random.uniform(-rotation, rotation)
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                image = cv2.warpAffine(
                    image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0
                )

                # Synchronous rotation key points
                visible_mask = keypoints[:, 2] > 0
                if np.any(visible_mask):
                    coords = keypoints[visible_mask, :2]
                    # Homogeneous coordinate transformation
                    coords_homo = np.column_stack([coords, np.ones(len(coords))])
                    coords_rotated = coords_homo @ M.T
                    keypoints[visible_mask, :2] = coords_rotated

            # Random Scaling
            scale_range = self.augmentation.get("scale", [0.9, 1.1])
            scale_range = self.augmentation.get("scale", [0.8, 1.2])
            if isinstance(scale_range, list) and len(scale_range) == 2:
                scale = random.uniform(scale_range[0], scale_range[1])
                new_w, new_h = int(w * scale), int(h * scale)
                scaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

                # Crop or pad to original size
                image = np.zeros((h, w), dtype=np.float32)
                y_offset = (h - new_h) // 2
                x_offset = (w - new_w) // 2

                if scale > 1:  # Crop after zooming in
                    y_start = (new_h - h) // 2
                    x_start = (new_w - new_w) // 2
                    image = scaled[y_start : y_start + h, x_start : x_start + w]
                else:  # Center after zooming out
                    image[
                        max(0, y_offset) : min(h, y_offset + new_h),
                        max(0, x_offset) : min(w, x_offset + new_w),
                    ] = scaled

                # Synchronize zoom of key points
                keypoints[:, 0] = keypoints[:, 0] * scale + x_offset
                keypoints[:, 1] = keypoints[:, 1] * scale + y_offset

        # ===== Pixel-level transformation (without changing the geometric position) =====
        if self.mode == "train":
            # Brightness Adjustment (OpenCV: Beta Value Adjustment)
            brightness = self.augmentation.get("brightness", 0.2)
            if brightness > 0 and random.random() < 0.5:
                beta = random.uniform(-brightness * 255, brightness * 255)
                image = np.clip(image + beta, 0, 255).astype(np.float32)

            # Contrast Adjustment (OpenCV: Alpha Value Adjustment)
            contrast = self.augmentation.get("contrast", 0.2)
            if contrast > 0 and random.random() < 0.5:
                alpha = random.uniform(1 - contrast, 1 + contrast)
                image = np.clip(image * alpha, 0, 255).astype(np.float32)

            # Gaussian Noise (for Training Set Only)
            if self.mode == "train":
                noise_scale = self.augmentation.get("gaussian_noise", 0.03)
                if noise_scale > 0 and random.random() < 0.3:
                    noise = np.random.normal(0, noise_scale * 255, image.shape)
                    image = np.clip(image + noise, 0, 255).astype(np.float32)
            # Stronger blur
            blur_prob = self.augmentation.get("blur_prob", 0.5)
            if random.random() < blur_prob:
                ksize = self.augmentation.get("blur_kernel", 5) | 1
                sigma = random.uniform(0.5, 2.0)
                image = cv2.GaussianBlur(image, (ksize, ksize), sigma)
        return image, keypoints

    def load_image(self, img_path):
        """Load image and preprocess"""
        # Read in grayscale mode
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"""Unable to load image: {img_path}""")

        # Keep as float32 for subsequent enhancement
        img = img.astype(np.float32)

        return img

    def load_keypoints(self, img_path):
        """Load keypoint annotations from Labelme JSON
        Returns: numpy array [num_keypoints, 3] -> (x, y, visibility)
        visibility: 0=invisible, 1=visible but obstructed, 2=visible"""
        keypoints = np.zeros((self.num_keypoints, 3), dtype=np.float32)

        if self.anno_dir is None:
            return keypoints

        # Find the corresponding JSON file
        json_path = self.anno_dir / f"{img_path.stem}.json"

        if not json_path.exists():
            # If there is no annotation, return all 0s (indicating no white dots).
            return keypoints

        try:
            # Get the actual image size
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                return keypoints
            actual_height, actual_width = img.shape

            with open(json_path, "r", encoding="utf-8-sig") as f:
                data = json.load(f)

            # Get the original image size in the annotation
            anno_width = data.get("imageWidth", actual_width)
            anno_height = data.get("imageHeight", actual_height)

            # Scale coordinates based on actual image dimensions
            scale_x = actual_width / anno_width
            scale_y = actual_height / anno_height

            # Define the keypoint order mapping
            if self.num_keypoints == 6:
                keypoint_order = {
                    "dot_1": 0,
                    "dot_2": 1,
                    "dot_3": 2,
                    "dot_4": 3,
                    "dot_5": 4,
                    "dot_6": 5,
                }
            else:
                keypoint_order = {
                    "left_dot_1": 0,
                    "left_dot_2": 1,
                    "right_dot_1": 2,
                    "right_dot_2": 3,
                }

            # Parse keypoints
            for shape in data.get("shapes", []):
                label = shape.get("label", "").lower()

                # Only read white dot annotations
                if label in keypoint_order:
                    kp_idx = keypoint_order[label]
                    points = shape.get("points", [])
                    if len(points) > 0:
                        x, y = points[0]
                        # Check for missing markers [-1, -1]
                        if x == -1 and y == -1:
                            keypoints[kp_idx, 0] = 0
                            keypoints[kp_idx, 1] = 0
                            keypoints[kp_idx, 2] = 0  # Not visible (missing)
                        else:
                            keypoints[kp_idx, 0] = x * scale_x
                            keypoints[kp_idx, 1] = y * scale_y
                            keypoints[kp_idx, 2] = 2  # Visible

        except Exception as e:
            print(f"""Failed to load annotation {json_path} : {e}""")

        return keypoints

    def get_num_dots(self, idx):
        """Get the number of white dots in the specified sample"""
        keypoints = self.load_keypoints(self.image_files[idx])
        return int(np.sum(keypoints[:, 2] > 0))


def get_data_loaders(config):
    """Create Data Loader (OpenCV Optimized Version)

    Args:
    config: Configuration dictionary

    Returns:
    train_loader, val_loader"""
    data_cfg = config["data"]

    # Get training and validation subdirectories
    train_subdir = data_cfg.get("train_subdir", "train")
    val_subdir = data_cfg.get("val_subdir", "val")

    # Annotation directory: Use train_aug if it exists; otherwise, use train.
    anno_subdir = (
        train_subdir
        if os.path.exists(os.path.join(data_cfg["anno_dir"], train_subdir))
        else train_subdir.replace("_aug", "")
    )

    # Get augmentation configuration
    aug_cfg = config.get("augmentation", {})

    train_dataset = DNASegmentationDataset(
        image_dir=os.path.join(data_cfg["processed_dir"], train_subdir),
        anno_dir=os.path.join(data_cfg["anno_dir"], anno_subdir),
        heatmap_dir=os.path.join(data_cfg["heatmap_dir"], train_subdir)
        if os.path.exists(os.path.join(data_cfg["heatmap_dir"], train_subdir))
        else None,
        input_size=tuple(data_cfg["input_size"]),
        num_keypoints=data_cfg["num_keypoints"],
        mode="train",
        augmentation=aug_cfg,
    )

    # Validation dataset
    val_dataset = DNASegmentationDataset(
        image_dir=os.path.join(data_cfg["processed_dir"], val_subdir),
        anno_dir=os.path.join(data_cfg["anno_dir"], val_subdir),
        heatmap_dir=os.path.join(data_cfg["heatmap_dir"], val_subdir)
        if os.path.exists(os.path.join(data_cfg["heatmap_dir"], val_subdir))
        else None,
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
    """Custom collate function to handle data of different sizes"""
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

    # Merge heatmaps if available
    if "heatmap" in batch[0]:
        heatmaps = torch.stack([item["heatmap"] for item in batch])
        result["heatmap"] = heatmaps

    return result


if __name__ == "__main__":
    # Test data set
    import time

    dataset = DNASegmentationDataset(
        image_dir="../data/processed/train",
        anno_dir="../data/annotations/train",
        input_size=(256, 256),
        num_keypoints=4,
        mode="train",
        augmentation={"rotation": 15, "flip_lr": 0.5, "brightness": 0.2, "contrast": 0.2},
    )

    print(f"""Dataset size: {len(dataset)}""")

    # Test loading speed
    start = time.time()
    for i in range(min(100, len(dataset))):
        sample = dataset[i]
    elapsed = time.time() - start
    print(
        f"""Time taken to load 100 images: {elapsed:.2f} s (average {elapsed / 100 * 1000:.1f} ms/image)"""
    )

    # Test loading a sample
    sample = dataset[0]
    print(f"""Image shape: {sample["image"].shape}""")
    print(f"""Key points: {sample["keypoints"]}""")
    print(f"""Number of white dots: {dataset.get_num_dots(0)}""")
