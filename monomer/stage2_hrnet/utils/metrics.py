""""""

import numpy as np
import torch
from collections import defaultdict


def compute_pck(pred_keypoints, gt_keypoints, image_size, threshold=0.05):
    """Percentage of Correct Keypoints (PCK)

    Args:
    pred_keypoints: [N, num_keypoints, 3] (x, y, visibility)
    gt_keypoints: [N, num_keypoints, 3] GT
    image_size: (H, W)
    threshold: ()

    Returns:
    pck: PCK (0-1)"""
    H, W = image_size
    norm_factor = np.sqrt(H**2 + W**2)  #

    correct = 0
    total = 0

    for pred, gt in zip(pred_keypoints, gt_keypoints):
        for i in range(len(pred)):
            #
            if gt[i, 2] == 0:
                continue

            total += 1

            #
            dist = np.sqrt((pred[i, 0] - gt[i, 0]) ** 2 + (pred[i, 1] - gt[i, 1]) ** 2)

            #
            if dist < threshold * norm_factor:
                correct += 1

    return correct / max(total, 1)


def compute_oks(pred_keypoints, gt_keypoints, image_size, sigmas=None):
    """Object Keypoint Similarity (OKS) - COCO

    Args:
    pred_keypoints: [N, num_keypoints, 3]
    gt_keypoints: [N, num_keypoints, 3]
    image_size: (H, W)
    sigmas: sigma

    Returns:
    oks: OKS"""
    N = len(pred_keypoints)
    H, W = image_size
    area = H * W

    if sigmas is None:
        sigmas = np.ones(pred_keypoints[0].shape[0]) * 0.1

    all_oks = []

    for pred, gt in zip(pred_keypoints, gt_keypoints):
        oks = 0
        num_visible = 0

        for i in range(pred.shape[0]):
            if gt[i, 2] == 0:  # GT
                continue

            num_visible += 1

            dx = pred[i, 0] - gt[i, 0]
            dy = pred[i, 1] - gt[i, 1]
            dist = np.sqrt(dx**2 + dy**2)

            var = (sigmas[i] * 2) ** 2
            oks += np.exp(-(dist**2) / (2 * var * area))

        if num_visible > 0:
            all_oks.append(oks / num_visible)

    return np.mean(all_oks) if all_oks else 0


def compute_ap(pred_keypoints, gt_keypoints, image_size, oks_thresholds=None):
    """Average Precision (AP) OKS

    Args:
    pred_keypoints:
    gt_keypoints: GT
    image_size:
    oks_thresholds: OKS

    Returns:
    ap_dict: AP"""
    if oks_thresholds is None:
        oks_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    # OKS
    oks_list = []
    for pred, gt in zip(pred_keypoints, gt_keypoints):
        oks = compute_oks([pred], [gt], image_size)
        oks_list.append(oks)

    # AP
    ap_dict = {}
    for thresh in oks_thresholds:
        correct = sum(oks >= thresh for oks in oks_list)
        ap_dict[f"AP@{thresh}"] = correct / max(len(oks_list), 1)

    # mAP()
    ap_dict["mAP"] = np.mean(list(ap_dict.values()))

    return ap_dict


def compute_classification_accuracy(pred_counts, gt_counts):
    """Args:
    pred_counts: [N]
    gt_counts: GT [N]

    Returns:
    accuracy:
    per_class_acc:"""
    pred_counts = np.array(pred_counts)
    gt_counts = np.array(gt_counts)

    #
    accuracy = np.mean(pred_counts == gt_counts)

    #
    unique_classes = np.unique(gt_counts)
    per_class_acc = {}

    for cls in unique_classes:
        mask = gt_counts == cls
        if mask.sum() > 0:
            per_class_acc[f"class_{cls}"] = np.mean(pred_counts[mask] == gt_counts[mask])
        else:
            per_class_acc[f"class_{cls}"] = 0.0

    return accuracy, per_class_acc


def compute_precision_recall_f1(pred_keypoints, gt_keypoints, distance_threshold=10):
    """Precision, Recall, F1

    Args:
    pred_keypoints: [N, num_keypoints, 3]
    gt_keypoints: [N, num_keypoints, 3]
    distance_threshold: ()

    Returns:
    precision, recall, f1"""
    tp = 0  # True Positive
    fp = 0  # False Positive
    fn = 0  # False Negative

    for pred, gt in zip(pred_keypoints, gt_keypoints):
        #
        pred_visible = pred[pred[:, 2] > 0][:, :2]
        gt_visible = gt[gt[:, 2] > 0][:, :2]

        if len(gt_visible) == 0:
            fp += len(pred_visible)
            continue

        if len(pred_visible) == 0:
            fn += len(gt_visible)
            continue

        #
        matched_gt = set()
        for pred_pt in pred_visible:
            # GT
            dists = np.sqrt(np.sum((gt_visible - pred_pt) ** 2, axis=1))
            min_dist_idx = np.argmin(dists)
            min_dist = dists[min_dist_idx]

            if min_dist < distance_threshold and min_dist_idx not in matched_gt:
                tp += 1
                matched_gt.add(min_dist_idx)
            else:
                fp += 1

        fn += len(gt_visible) - len(matched_gt)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return precision, recall, f1


class MetricsTracker:
    """"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.metrics = defaultdict(list)
        self.epoch_metrics = {}

    def update(self, **kwargs):
        """"""
        for key, value in kwargs.items():
            self.metrics[key].append(value)

    def get_average(self, key):
        """"""
        if key in self.metrics:
            return np.mean(self.metrics[key])
        return 0

    def get_current(self, key):
        """"""
        if key in self.metrics and len(self.metrics[key]) > 0:
            return self.metrics[key][-1]
        return 0

    def set_epoch_metrics(self, epoch, **kwargs):
        """epoch"""
        self.epoch_metrics[epoch] = kwargs

    def get_best_epoch(self, metric="val_loss", mode="min"):
        """epoch"""
        if not self.epoch_metrics:
            return None

        epochs = list(self.epoch_metrics.keys())
        values = [
            self.epoch_metrics[e].get(metric, float("inf") if mode == "min" else float("-inf"))
            for e in epochs
        ]

        if mode == "min":
            best_idx = np.argmin(values)
        else:
            best_idx = np.argmax(values)

        return epochs[best_idx], values[best_idx]

    def summary(self):
        """"""
        print("\n" + "=" * 50)
        print("""""")
        print("=" * 50)

        for key in self.metrics:
            avg = self.get_average(key)
            current = self.get_current(key)
            print(f"""{key} : = {avg:.4f} , = {current:.4f}""")

        print("=" * 50)


def test_metrics():
    """"""
    #
    pred_kp = np.array(
        [
            [[50, 50, 2], [100, 100, 2], [0, 0, 0], [0, 0, 0]],  # 2
            [[60, 55, 2], [0, 0, 0], [0, 0, 0], [0, 0, 0]],  # 1
        ]
    )

    gt_kp = np.array(
        [
            [[52, 48, 2], [98, 102, 2], [0, 0, 0], [0, 0, 0]],
            [[58, 52, 2], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
        ]
    )

    image_size = (256, 256)

    #
    pck = compute_pck(pred_kp, gt_kp, image_size, threshold=0.05)
    oks = compute_oks(pred_kp, gt_kp, image_size)
    precision, recall, f1 = compute_precision_recall_f1(pred_kp, gt_kp)

    print(f"PCK: {pck:.4f}")
    print(f"OKS: {oks:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")

    #
    pred_counts = [2, 1]
    gt_counts = [2, 1]
    acc, per_class = compute_classification_accuracy(pred_counts, gt_counts)
    print(f"\nClassification Accuracy: {acc:.4f}")
    print(f"Per-class Accuracy: {per_class}")


if __name__ == "__main__":
    test_metrics()
