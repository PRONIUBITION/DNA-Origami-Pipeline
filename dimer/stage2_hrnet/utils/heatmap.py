"""Heatmap Generation Tool
Convert keypoint coordinates to a Gaussian heatmap"""

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import cv2


def generate_heatmap(keypoints, output_size, sigma=3, num_keypoints=4):
    """Generate a Gaussian heatmap from keypoints

    Args:
    keypoints: NumPy array [num_keypoints, 3] (x, y, visibility)
    output_size: (H, W) Heatmap size
    sigma: Gaussian kernel standard deviation
    num_keypoints: Number of keypoints

    Returns:
    heatmaps: NumPy array [num_keypoints, H, W]"""
    H, W = output_size
    heatmaps = np.zeros((num_keypoints, H, W), dtype=np.float32)

    for i in range(num_keypoints):
        x, y, visibility = keypoints[i]

        # A visibility of 0 indicates that the keypoint does not exist (no small white dot).
        if visibility == 0:
            continue

        # Ensure that the coordinates are within the range.
        x = int(np.clip(x, 0, W - 1))
        y = int(np.clip(y, 0, H - 1))

        # Generate a Gaussian distribution
        tmp_size = sigma * 3
        mu_x = x
        mu_y = y

        # Check boundaries
        x0 = int(max(0, mu_x - tmp_size))
        y0 = int(max(0, mu_y - tmp_size))
        x1 = int(min(W, mu_x + tmp_size + 1))
        y1 = int(min(H, mu_y + tmp_size + 1))

        # Generate grid
        xs = np.arange(x0, x1, 1, np.float32)
        ys = np.arange(y0, y1, 1, np.float32)
        xs = xs[np.newaxis, :].repeat(y1 - y0, axis=0)
        ys = ys[:, np.newaxis].repeat(x1 - x0, axis=1)

        # Calculate Gaussian value
        gaussian = np.exp(-((xs - mu_x) ** 2 + (ys - mu_y) ** 2) / (2 * sigma**2))

        # Assign to heat map
        heatmaps[i, y0:y1, x0:x1] = np.maximum(heatmaps[i, y0:y1, x0:x1], gaussian)

    return heatmaps


def batch_generate_heatmaps(dataset, output_dir, sigma=3):
    """Batch generation of heatmaps for datasets

    Args:
    dataset: DNASegmentationDataset instance
    output_dir: Output directory
    sigma: Gaussian kernel standard deviation"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"""Starting to generate heatmaps, with a total of {len(dataset)} samples...""")

    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        keypoints = sample["keypoints"]
        image_name = sample["image_name"]

        # Get image dimensions
        image = sample["image"]
        if isinstance(image, torch.Tensor):
            H, W = image.shape[1], image.shape[2]
        else:
            H, W = image.shape[:2]

        # The coordinates of the key points need to be scaled according to the output size of the heat map.
        # Assuming the heat map is 1/4 of the input size
        output_H, output_W = H // 4, W // 4

        # Scale the keypoint coordinates
        keypoints_scaled = keypoints.copy()
        keypoints_scaled[:, 0] = keypoints[:, 0] * (output_W / W)
        keypoints_scaled[:, 1] = keypoints[:, 1] * (output_H / H)

        # Generate a heatmap
        heatmaps = generate_heatmap(
            keypoints_scaled,
            output_size=(output_H, output_W),
            sigma=sigma,
            num_keypoints=dataset.num_keypoints,
        )

        # Save Heatmap
        save_path = output_dir / f"{image_name}_heatmap.npy"
        np.save(save_path, heatmaps)

    print(f"""Heatmap generation completed and saved in: {output_dir}""")


def heatmap_to_coordinates(heatmap, threshold=0.5, nms_distance=5, cross_channel_nms=True):
    """Extract keypoint coordinates from the heatmap (with cross-channel NMS)

    Args:
    heatmap: numpy array [H, W] or [num_keypoints, H, W]
    threshold: peak threshold
    nms_distance: Cross-channel NMS distance threshold (pixels)

    Returns:
    coords: list of (x, y, confidence)"""
    if len(heatmap.shape) == 2:
        heatmap = heatmap[np.newaxis, ...]

    num_keypoints, H, W = heatmap.shape

    # Step 1: Collect the peak values from all channels
    all_peaks = []  # (channel_idx, x, y, confidence)
    for i in range(num_keypoints):
        hm = heatmap[i]
        y, x = np.unravel_index(np.argmax(hm), hm.shape)
        confidence = hm[y, x]

        if confidence > threshold:
            all_peaks.append((i, x, y, confidence))

    # Step 2: Sort in descending order by confidence level
    all_peaks.sort(key=lambda x: x[3], reverse=True)

    if not cross_channel_nms:
        coords = [(-1, -1, 0) for _ in range(num_keypoints)]
        used_positions = set()
        for ch_idx, x, y, conf in all_peaks:
            pos = (int(x), int(y))
            if pos not in used_positions:
                coords[ch_idx] = (x, y, conf)
                used_positions.add(pos)
        return coords

    # Step 3: Apply NMS and select peaks
    selected_peaks = {}  # channel_idx -> (x, y, confidence)
    selected_positions = []  # (x, y) for NMS check

    for ch_idx, x, y, conf in all_peaks:
        # Check if it is too close to the selected points
        is_valid = True
        for prev_x, prev_y in selected_positions:
            dist = np.sqrt((x - prev_x) ** 2 + (y - prev_y) ** 2)
            if dist < nms_distance:
                is_valid = False
                break

        if is_valid:
            selected_peaks[ch_idx] = (x, y, conf)
            selected_positions.append((x, y))

    # Step 4: Construct the result (maintain channel order)
    coords = []
    for i in range(num_keypoints):
        if i in selected_peaks:
            x, y, conf = selected_peaks[i]
            coords.append((x, y, conf))
        else:
            coords.append((-1, -1, 0))  # Not detected or filtered by NMS

    return coords


def apply_nms(heatmap, kernel_size=3):
    """Apply Non-Maximum Suppression (NMS) to the heatmap

    Args:
    heatmap: [num_keypoints, H, W]
    kernel_size: NMS kernel size

    Returns:
    Heatmap after NMS"""
    if isinstance(heatmap, np.ndarray):
        heatmap = torch.from_numpy(heatmap)

    num_keypoints, H, W = heatmap.shape

    # Use maxpool for NMS
    pad = (kernel_size - 1) // 2
    hmax = F.max_pool2d(
        heatmap.unsqueeze(0), kernel_size=kernel_size, stride=1, padding=pad
    ).squeeze(0)

    keep = (hmax == heatmap).float()
    heatmap = heatmap * keep

    return heatmap.numpy() if isinstance(heatmap, np.ndarray) else heatmap


def visualize_heatmap_overlay(image, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """Overlay the heatmap on the original image for visualization

    Args:
    image: Original image [H, W] or [H, W, 3]
    heatmap: Single-channel heatmap [H, W]
    alpha: Overlay transparency
    colormap: OpenCV color map

    Returns:
    Overlayed image"""
    # Ensure that the image is of the uint8 type
    if isinstance(image, torch.Tensor):
        image = image.numpy()
    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.numpy()

    # Normalize the image
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)

    # Ensure it is 3-channel
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Normalized heat map
    heatmap = np.clip(heatmap, 0, 1)
    heatmap = (heatmap * 255).astype(np.uint8)

    # Apply Color Mapping
    heatmap_color = cv2.applyColorMap(heatmap, colormap)

    # Overlay
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap_color, alpha, 0)

    return overlay


def compute_oks(pred_keypoints, gt_keypoints, image_size, sigmas=None):
    """Calculate Object Keypoint Similarity (OKS)

    Args:
    pred_keypoints: [num_keypoints, 3] (x, y, visibility)
    gt_keypoints: [num_keypoints, 3]
    image_size: (H, W)
    sigmas: Standard deviation for each keypoint

    Returns:
    oks: OKS score"""
    if sigmas is None:
        # Use default sigma
        sigmas = np.ones(pred_keypoints.shape[0]) * 0.1

    H, W = image_size
    area = H * W

    oks = 0
    num_visible = 0

    for i in range(pred_keypoints.shape[0]):
        if gt_keypoints[i, 2] == 0:  # GT is not visible
            continue

        num_visible += 1

        # Calculate Euclidean distance
        dx = pred_keypoints[i, 0] - gt_keypoints[i, 0]
        dy = pred_keypoints[i, 1] - gt_keypoints[i, 1]
        dist = np.sqrt(dx**2 + dy**2)

        # Calculate OKS
        var = (sigmas[i] * 2) ** 2
        oks += np.exp(-(dist**2) / (2 * var * area))

    return oks / max(num_visible, 1)


if __name__ == "__main__":
    # Test heat map generation
    keypoints = np.array(
        [
            [50, 50, 2],  # Small white dot 1
            [100, 80, 2],  # Small white dot 2
            [0, 0, 0],  # No white dot 3
            [0, 0, 0],  # No white dot 4
        ],
        dtype=np.float32,
    )

    heatmaps = generate_heatmap(keypoints, output_size=(64, 64), sigma=3)
    print(f"""Heatmap shape: {heatmaps.shape}""")

    # Test extracting coordinates from the heat map
    coords = heatmap_to_coordinates(heatmaps, threshold=0.3)
    print(f"""Extracted coordinates: {coords}""")

    # Visualize Heatmap
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    for i in range(4):
        axes[i].imshow(heatmaps[i], cmap="hot")
        axes[i].set_title(f"Keypoint {i + 1}")
        axes[i].axis("off")
    plt.tight_layout()
    plt.savefig("test_heatmaps.png")
    print("""The heatmap has been saved to test_heatmaps.png""")
