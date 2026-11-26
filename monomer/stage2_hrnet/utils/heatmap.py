"""?"""

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import cv2


def generate_heatmap(keypoints, output_size, sigma=3, num_keypoints=4):
    """?

    Args:
    keypoints: numpy array [num_keypoints, 3] (x, y, visibility)
    output_size: (H, W) ?
    sigma:
    num_keypoints: ?

    Returns:
    heatmaps: numpy array [num_keypoints, H, W]"""
    H, W = output_size
    heatmaps = np.zeros((num_keypoints, H, W), dtype=np.float32)

    for i in range(num_keypoints):
        x, y, visibility = keypoints[i]

        # visibility?C
        if visibility == 0:
            continue

        #
        x = int(np.clip(x, 0, W - 1))
        y = int(np.clip(y, 0, H - 1))

        #
        tmp_size = sigma * 3
        mu_x = x
        mu_y = y

        # ?
        x0 = int(max(0, mu_x - tmp_size))
        y0 = int(max(0, mu_y - tmp_size))
        x1 = int(min(W, mu_x + tmp_size + 1))
        y1 = int(min(H, mu_y + tmp_size + 1))

        #
        xs = np.arange(x0, x1, 1, np.float32)
        ys = np.arange(y0, y1, 1, np.float32)
        xs = xs[np.newaxis, :].repeat(y1 - y0, axis=0)
        ys = ys[:, np.newaxis].repeat(x1 - x0, axis=1)

        # $?
        gaussian = np.exp(-((xs - mu_x) ** 2 + (ys - mu_y) ** 2) / (2 * sigma**2))

        # ?
        heatmaps[i, y0:y1, x0:x1] = np.maximum(heatmaps[i, y0:y1, x0:x1], gaussian)

    return heatmaps


def batch_generate_heatmaps(dataset, output_dir, sigma=3):
    """?

    Args:
    dataset: DNASegmentationDataset
    output_dir:
    sigma:"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"""{len(dataset)} ?..""")

    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        keypoints = sample["keypoints"]
        image_name = sample["image_name"]

        #
        image = sample["image"]
        if isinstance(image, torch.Tensor):
            H, W = image.shape[1], image.shape[2]
        else:
            H, W = image.shape[:2]

        #
        # ?/4
        output_H, output_W = H // 4, W // 4

        # ?
        keypoints_scaled = keypoints.copy()
        keypoints_scaled[:, 0] = keypoints[:, 0] * (output_W / W)
        keypoints_scaled[:, 1] = keypoints[:, 1] * (output_H / H)

        # ?
        heatmaps = generate_heatmap(
            keypoints_scaled,
            output_size=(output_H, output_W),
            sigma=sigma,
            num_keypoints=dataset.num_keypoints,
        )

        # ?
        save_path = output_dir / f"{image_name}_heatmap.npy"
        np.save(save_path, heatmaps)

    print(f"""? {output_dir}""")


def heatmap_to_coordinates(heatmap, threshold=0.5, nms_distance=5):
    """NMS?

    Args:
    heatmap: numpy array [H, W] ?[num_keypoints, H, W]
    threshold: ?
    nms_distance: NMS?

    Returns:
    coords: list of (x, y, confidence)"""
    if len(heatmap.shape) == 2:
        heatmap = heatmap[np.newaxis, ...]

    num_keypoints, H, W = heatmap.shape

    # ?
    all_peaks = []  # (channel_idx, x, y, confidence)
    for i in range(num_keypoints):
        hm = heatmap[i]
        y, x = np.unravel_index(np.argmax(hm), hm.shape)
        confidence = hm[y, x]

        if confidence > threshold:
            all_peaks.append((i, x, y, confidence))

    #
    all_peaks.sort(key=lambda x: x[3], reverse=True)

    # NMS?
    selected_peaks = {}  # channel_idx -> (x, y, confidence)
    selected_positions = []  # (x, y) for NMS check

    for ch_idx, x, y, conf in all_peaks:
        #
        is_valid = True
        for prev_x, prev_y in selected_positions:
            dist = np.sqrt((x - prev_x) ** 2 + (y - prev_y) ** 2)
            if dist < nms_distance:
                is_valid = False
                break

        if is_valid:
            selected_peaks[ch_idx] = (x, y, conf)
            selected_positions.append((x, y))

    # ?
    coords = []
    for i in range(num_keypoints):
        if i in selected_peaks:
            x, y, conf = selected_peaks[i]
            coords.append((x, y, conf))
        else:
            coords.append((-1, -1, 0))  # NMS

    return coords


def apply_nms(heatmap, kernel_size=3):
    """?NMS)

    Args:
    heatmap: [num_keypoints, H, W]
    kernel_size: NMS?

    Returns:
    nms?"""
    if isinstance(heatmap, np.ndarray):
        heatmap = torch.from_numpy(heatmap)

    num_keypoints, H, W = heatmap.shape

    # maxpoolNMS
    pad = (kernel_size - 1) // 2
    hmax = F.max_pool2d(
        heatmap.unsqueeze(0), kernel_size=kernel_size, stride=1, padding=pad
    ).squeeze(0)

    keep = (hmax == heatmap).float()
    heatmap = heatmap * keep

    return heatmap.numpy() if isinstance(heatmap, np.ndarray) else heatmap


def visualize_heatmap_overlay(image, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """?

    Args:
    image: [H, W] ?[H, W, 3]
    heatmap: ?[H, W]
    alpha: ?
    colormap: OpenCV

    Returns:"""
    # int8
    if isinstance(image, torch.Tensor):
        image = image.numpy()
    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.numpy()

    # ?
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)

    # ?
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    #
    heatmap = np.clip(heatmap, 0, 1)
    heatmap = (heatmap * 255).astype(np.uint8)

    #
    heatmap_color = cv2.applyColorMap(heatmap, colormap)

    #
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap_color, alpha, 0)

    return overlay


def compute_oks(pred_keypoints, gt_keypoints, image_size, sigmas=None):
    """$Object Keypoint Similarity (OKS)

    Args:
    pred_keypoints: [num_keypoints, 3] (x, y, visibility)
    gt_keypoints: [num_keypoints, 3]
    image_size: (H, W)
    sigmas: ?

    Returns:
    oks: OKS"""
    if sigmas is None:
        # sigma
        sigmas = np.ones(pred_keypoints.shape[0]) * 0.1

    H, W = image_size
    area = H * W

    oks = 0
    num_visible = 0

    for i in range(pred_keypoints.shape[0]):
        if gt_keypoints[i, 2] == 0:  # GT?
            continue

        num_visible += 1

        # $
        dx = pred_keypoints[i, 0] - gt_keypoints[i, 0]
        dy = pred_keypoints[i, 1] - gt_keypoints[i, 1]
        dist = np.sqrt(dx**2 + dy**2)

        # $OKS
        var = (sigmas[i] * 2) ** 2
        oks += np.exp(-(dist**2) / (2 * var * area))

    return oks / max(num_visible, 1)


if __name__ == "__main__":
    # ?
    keypoints = np.array(
        [
            [50, 50, 2],  # ?
            [100, 80, 2],  # ?
            [0, 0, 0],  # 3
            [0, 0, 0],  # 4
        ],
        dtype=np.float32,
    )

    heatmaps = generate_heatmap(keypoints, output_size=(64, 64), sigma=3)
    print(f"""? {heatmaps.shape}""")

    #
    coords = heatmap_to_coordinates(heatmaps, threshold=0.3)
    print(f"""? {coords}""")

    #
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    for i in range(4):
        axes[i].imshow(heatmaps[i], cmap="hot")
        axes[i].set_title(f"Keypoint {i + 1}")
        axes[i].axis("off")
    plt.tight_layout()
    plt.savefig("test_heatmaps.png")
    print("""?test_heatmaps.png""")


def generate_heatmap_gpu(keypoints, heatmap_size, sigma, num_keypoints):
    import torch

    h, w = heatmap_size
    kps = keypoints if torch.is_tensor(keypoints) else torch.from_numpy(keypoints).float()
    kps = kps.to("cuda")
    hm = torch.zeros((num_keypoints, h, w), device="cuda")
    ys, xs = torch.meshgrid(
        torch.arange(h, device="cuda"), torch.arange(w, device="cuda"), indexing="ij"
    )
    for i in range(num_keypoints):
        x, y, v = kps[i]
        if v > 0 and x >= 0 and y >= 0:
            hm[i] = torch.exp(-((xs - x) ** 2 + (ys - y) ** 2) / (2 * sigma**2))
    return hm.cpu().numpy()
