import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

import predict as hrnet_predict
from utils import heatmap_to_coordinates


COLORS = [
    (0, 0, 255),
    (0, 165, 255),
    (0, 255, 255),
    (0, 255, 0),
    (255, 0, 0),
    (255, 0, 255),
]

LABELS = ["dot_1", "dot_2", "dot_3", "dot_4", "dot_5", "dot_6"]


def run_on_directory(model, image_dir, config, output_dir, device):
    input_size = tuple(config["data"]["input_size"])
    heatmap_size = tuple(config["heatmap"]["output_size"])
    num_keypoints = config["data"]["num_keypoints"]
    threshold = config["inference"]["heatmap_threshold"]
    nms_distance = config["inference"].get("nms_distance", 5)

    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(
        p
        for p in image_dir.iterdir()
        if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff")
    )

    results = []
    for image_path in image_files:
        img_tensor, original_size, original_img = hrnet_predict.preprocess_image(
            image_path, input_size
        )
        img_tensor = img_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            heatmaps = model(img_tensor)[0].cpu().numpy()

        coords = heatmap_to_coordinates(
            heatmaps,
            threshold=threshold,
            nms_distance=nms_distance,
            cross_channel_nms=False,
        )

        keypoints = np.zeros((num_keypoints, 3), dtype=np.float32)
        scale_x = original_size[1] / input_size[1]
        scale_y = original_size[0] / input_size[0]

        for idx, (x, y, conf) in enumerate(coords):
            if conf <= 0:
                continue
            input_x = x * (input_size[1] / heatmap_size[1])
            input_y = y * (input_size[0] / heatmap_size[0])
            keypoints[idx, 0] = input_x * scale_x
            keypoints[idx, 1] = input_y * scale_y
            keypoints[idx, 2] = conf

        result = {
            "image_name": image_path.stem,
            "image_path": str(image_path),
            "original_size": list(original_size),
            "keypoints": keypoints.tolist(),
            "num_dots": int(np.sum(keypoints[:, 2] > threshold)),
        }
        results.append(result)

        vis = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
        h, w = vis.shape[:2]
        for idx, (x, y, conf) in enumerate(keypoints):
            if conf <= threshold:
                continue
            color = COLORS[idx % len(COLORS)]
            cv2.circle(vis, (int(round(x)), int(round(y))), 1, color, -1)

        cv2.imwrite(str(viz_dir / f"{image_path.stem}_pred.png"), vis)

    with open(output_dir / "predictions.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config_fourbits_overfit.yaml")
    parser.add_argument("--weights", required=True)
    parser.add_argument(
        "--input",
        default="data/fourbits/test_set",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device(args.device)
    print(f"Loading weights: {args.weights}")
    model = hrnet_predict.load_model(args.weights, config, device)
    model.eval()

    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("outputs") / "fourbits_test" / timestamp
    else:
        output_dir = Path(args.output)

    results = run_on_directory(model, args.input, config, output_dir, device)

    from collections import Counter

    counts = Counter(r["num_dots"] for r in results)
    print(f"Predicted {len(results)} images")
    print("Dot count distribution:")
    for count in sorted(counts):
        print(f"  {count} dots: {counts[count]}")
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
