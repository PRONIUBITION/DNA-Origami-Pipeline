"""Visualization Tools
Used to visualize prediction results and the training process"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from pathlib import Path


def visualize_keypoints(image, keypoints, save_path=None, title="Keypoints"):
    """Visualize keypoints on the image

    Args:
    image: Input image [H, W] or [H, W, 3]
    keypoints: [num_keypoints, 3] (x, y, visibility)
    save_path: Save path
    title: Image title"""
    # Prepare the image
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if len(image.shape) == 3 and image.shape[0] in [1, 3]:
        image = image.transpose(1, 2, 0)
    if image.shape[-1] == 1:
        image = image.squeeze(-1)

    # Normalize to 0-255
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)

    # Convert to 3 channels
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Copy for drawing
    vis_img = image.copy()

    # Draw keypoints
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]  # Red, Green, Blue, Yellow

    for i, (x, y, vis) in enumerate(keypoints):
        if vis > 0:  # Visible
            cx, cy = int(x), int(y)
            color = colors[i % len(colors)]

            # Draw a circle
            cv2.circle(vis_img, (cx, cy), 5, color, -1)
            cv2.circle(vis_img, (cx, cy), 7, (255, 255, 255), 2)

            # Annotation number
            cv2.putText(vis_img, str(i + 1), (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display
    plt.figure(figsize=(8, 8))
    plt.imshow(vis_img)
    plt.title(title)
    plt.axis("off")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"""Saved to: {save_path}""")

    plt.show()

    return vis_img


def visualize_predictions(
    image, pred_keypoints, gt_keypoints=None, save_path=None, title="Predictions"
):
    """Visualization of prediction results and comparison with GT

    Args:
    image: Input image
    pred_keypoints: Predicted keypoints [num_keypoints, 3]
    gt_keypoints: GT keypoints [num_keypoints, 3] (optional)
    save_path: Save path
    title: Title"""
    # Prepare the image
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if len(image.shape) == 3 and image.shape[0] in [1, 3]:
        image = image.transpose(1, 2, 0)
    if image.shape[-1] == 1:
        image = image.squeeze(-1)

    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Create a comparison image
    if gt_keypoints is not None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original image
        axes[0].imshow(image)
        axes[0].set_title("Original")
        axes[0].axis("off")

        # GT
        gt_img = image.copy()
        for i, (x, y, vis) in enumerate(gt_keypoints):
            if vis > 0:
                cv2.circle(gt_img, (int(x), int(y)), 5, (0, 255, 0), -1)
                cv2.putText(
                    gt_img,
                    str(i + 1),
                    (int(x) + 10, int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )
        axes[1].imshow(gt_img)
        axes[1].set_title(f"Ground Truth ({int(sum(gt_keypoints[:, 2] > 0))} dots)")
        axes[1].axis("off")

        # Forecast
        pred_img = image.copy()
        for i, (x, y, vis) in enumerate(pred_keypoints):
            if vis > 0:
                cv2.circle(pred_img, (int(x), int(y)), 5, (255, 0, 0), -1)
                cv2.putText(
                    pred_img,
                    str(i + 1),
                    (int(x) + 10, int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    2,
                )
        axes[2].imshow(pred_img)
        axes[2].set_title(f"Prediction ({int(sum(pred_keypoints[:, 2] > 0))} dots)")
        axes[2].axis("off")
    else:
        # Forecast only
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        pred_img = image.copy()
        for i, (x, y, vis) in enumerate(pred_keypoints):
            if vis > 0:
                cv2.circle(pred_img, (int(x), int(y)), 5, (255, 0, 0), -1)
                cv2.putText(
                    pred_img,
                    str(i + 1),
                    (int(x) + 10, int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    2,
                )
        ax.imshow(pred_img)
        ax.set_title(f"Prediction ({int(sum(pred_keypoints[:, 2] > 0))} dots)")
        ax.axis("off")

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"""Saved to: {save_path}""")

    plt.show()


def visualize_heatmap(image, heatmap, alpha=0.5, save_path=None):
    """Visualized heat map overlay

    Args:
    image: Original image
    heatmap: Heatmap [H, W] or [num_keypoints, H, W]
    alpha: Transparency
    save_path: Save path"""
    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.cpu().numpy()
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()

    # If it is a multi-channel heatmap, take the maximum value.
    if len(heatmap.shape) == 3:
        heatmap = np.max(heatmap, axis=0)

    # Prepare the image
    if len(image.shape) == 3 and image.shape[0] in [1, 3]:
        image = image.transpose(1, 2, 0)
    if image.shape[-1] == 1:
        image = image.squeeze(-1)

    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Normalized heat map
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    heatmap = (heatmap * 255).astype(np.uint8)

    # Apply color mapping
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # Resize the heatmap to match the original image
    if heatmap_color.shape[:2] != image.shape[:2]:
        heatmap_color = cv2.resize(heatmap_color, (image.shape[1], image.shape[0]))

    # Overlay
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap_color, alpha, 0)

    # Display
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image, cmap="gray" if len(image.shape) == 2 else None)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(heatmap_color)
    axes[1].set_title("Heatmap")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"""Saved to: {save_path}""")

    plt.show()


def visualize_batch(images, pred_keypoints, gt_keypoints=None, max_samples=4, save_path=None):
    """Batch Visualization

    Args:
    images: [B, C, H, W]
    pred_keypoints: list of [num_keypoints, 3]
    gt_keypoints: list of [num_keypoints, 3] (optional)
    max_samples: Maximum number of samples to display
    save_path: Save path"""
    batch_size = min(len(images), max_samples)

    fig, axes = plt.subplots(
        batch_size, 3 if gt_keypoints else 2, figsize=(12 if gt_keypoints else 8, 4 * batch_size)
    )

    if batch_size == 1:
        axes = axes.reshape(1, -1)

    for i in range(batch_size):
        image = images[i]
        pred_kp = pred_keypoints[i]

        # Prepare the image
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        if image.shape[0] in [1, 3]:
            image = image.transpose(1, 2, 0)
        if image.shape[-1] == 1:
            image = image.squeeze(-1)
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Original image
        axes[i, 0].imshow(image)
        axes[i, 0].set_title(f"Sample {i + 1}")
        axes[i, 0].axis("off")

        # GT
        if gt_keypoints:
            gt_kp = gt_keypoints[i]
            gt_img = image.copy()
            for j, (x, y, vis) in enumerate(gt_kp):
                if vis > 0:
                    cv2.circle(gt_img, (int(x), int(y)), 5, (0, 255, 0), -1)
            axes[i, 1].imshow(gt_img)
            axes[i, 1].set_title(f"GT ({int(sum(gt_kp[:, 2] > 0))} dots)")
            axes[i, 1].axis("off")

            # Forecast
            pred_img = image.copy()
            for j, (x, y, vis) in enumerate(pred_kp):
                if vis > 0:
                    cv2.circle(pred_img, (int(x), int(y)), 5, (255, 0, 0), -1)
            axes[i, 2].imshow(pred_img)
            axes[i, 2].set_title(f"Pred ({int(sum(pred_kp[:, 2] > 0))} dots)")
            axes[i, 2].axis("off")
        else:
            # Forecast only
            pred_img = image.copy()
            for j, (x, y, vis) in enumerate(pred_kp):
                if vis > 0:
                    cv2.circle(pred_img, (int(x), int(y)), 5, (255, 0, 0), -1)
            axes[i, 1].imshow(pred_img)
            axes[i, 1].set_title(f"Pred ({int(sum(pred_kp[:, 2] > 0))} dots)")
            axes[i, 1].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"""Saved to: {save_path}""")

    plt.show()


def plot_training_curves(train_losses, val_losses, metrics=None, save_path=None):
    """Plot the training curve

    Args:
    train_losses: List of training losses
    val_losses: List of validation losses
    metrics: Dictionary of other metrics
    save_path: Save path"""
    num_plots = 2 if metrics else 1
    fig, axes = plt.subplots(1, num_plots, figsize=(8 * num_plots, 5))

    if num_plots == 1:
        axes = [axes]

    # Loss curve
    epochs = range(1, len(train_losses) + 1)
    axes[0].plot(epochs, train_losses, "b-", label="Train Loss")
    axes[0].plot(epochs, val_losses, "r-", label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Other metrics
    if metrics:
        for name, values in metrics.items():
            axes[1].plot(range(1, len(values) + 1), values, label=name)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Value")
        axes[1].set_title("Metrics")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"""Saved to: {save_path}""")

    plt.show()


def visualize_confusion_matrix(confusion_mat, class_names, save_path=None):
    """Visualize the confusion matrix

    Args:
    confusion_mat: Confusion matrix [num_classes, num_classes]
    class_names: List of class names
    save_path: Save path"""
    plt.figure(figsize=(8, 6))
    plt.imshow(confusion_mat, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Add numerical labels
    thresh = confusion_mat.max() / 2.0
    for i, j in np.ndindex(confusion_mat.shape):
        plt.text(
            j,
            i,
            format(confusion_mat[i, j], "d"),
            ha="center",
            va="center",
            color="white" if confusion_mat[i, j] > thresh else "black",
        )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"""Saved to: {save_path}""")

    plt.show()


if __name__ == "__main__":
    # Test Visualization
    print("""Test visualization function...""")

    # Generate test data
    image = np.random.rand(256, 256).astype(np.float32)
    keypoints = np.array(
        [
            [100, 100, 2],
            [150, 120, 2],
            [0, 0, 0],
            [0, 0, 0],
        ]
    )

    visualize_keypoints(image, keypoints, title="Test Keypoints")
    print("""Test completed!""")
