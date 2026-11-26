"""HRNet -
DNA,"""

import os
import sys
import argparse
import yaml
import json
import shutil
import numpy as np
import torch
import cv2
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))

from models import HRNet, SimpleHRNet, hrnet_w32_cfg
from utils import heatmap_to_coordinates


def load_model(checkpoint_path, config, device="cuda"):
    """"""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model_type = config["model"].get("hrnet_type", "SimpleHRNet")
    num_keypoints = config["data"]["num_keypoints"]

    if model_type == "SimpleHRNet":
        model = SimpleHRNet(num_keypoints=num_keypoints, in_channels=1)
    elif model_type == "HRNet-W32":
        cfg = hrnet_w32_cfg.copy()
        cfg["NUM_HEATMAPS"] = num_keypoints
        model = HRNet(cfg)
    else:
        raise ValueError(f""": {model_type}""")

    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint.get("model")))
    model.to(device)
    model.eval()
    return model


def preprocess_image(image_path, input_size=(256, 256)):
    """"""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f""": {image_path}""")

    original_size = img.shape[:2]
    img_resized = cv2.resize(img, (input_size[1], input_size[0]))
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_normalized = (img_normalized - 0.5) / 0.5  # [-1,1]

    img_tensor = torch.from_numpy(img_normalized).unsqueeze(0).float()
    return img_tensor, original_size, img


def predict_single(model, image_path, config, device="cuda", nms_distance=None):
    """Returns:
    dict: {image_name, keypoints, num_dots, heatmap, original_size, expected_kp}"""
    input_size = tuple(config["data"]["input_size"])
    num_keypoints = config["data"]["num_keypoints"]
    heatmap_size = tuple(config["heatmap"]["output_size"])
    threshold = config["inference"]["heatmap_threshold"]
    expected_kp = config["data"].get("expected_keypoints", num_keypoints)

    if nms_distance is None:
        nms_distance = config["inference"].get("nms_distance", 5)

    img_tensor, original_size, original_img = preprocess_image(image_path, input_size)
    img_tensor = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        heatmaps_pred = model(img_tensor)

    heatmaps_pred = heatmaps_pred[0].cpu().numpy()
    pred_coords = heatmap_to_coordinates(
        heatmaps_pred, threshold=threshold, nms_distance=nms_distance
    )

    #
    pred_keypoints = np.zeros((num_keypoints, 3))
    scale_x = original_size[1] / input_size[1]
    scale_y = original_size[0] / input_size[0]

    for i, (x, y, conf) in enumerate(pred_coords):
        if conf > 0:
            input_x = x * (input_size[1] / heatmap_size[1])
            input_y = y * (input_size[0] / heatmap_size[0])
            pred_keypoints[i, 0] = input_x * scale_x
            pred_keypoints[i, 1] = input_y * scale_y
            pred_keypoints[i, 2] = conf

    num_dots = int(np.sum(pred_keypoints[:, 2] > threshold))

    return {
        "image_path": str(image_path),
        "image_name": Path(image_path).stem,
        "keypoints": pred_keypoints,
        "num_dots": num_dots,
        "heatmap": heatmaps_pred,
        "original_size": original_size,
        "expected_kp": expected_kp,
    }


def predict_batch(
    model,
    image_dir,
    config,
    device="cuda",
    output_dir=None,
    val_mode=False,
    checkpoint_path=None,
    nms_distance=None,
):
    """"""
    image_dir = Path(image_dir)
    image_files = sorted(
        [
            f
            for f in image_dir.iterdir()
            if f.suffix.lower() in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]
        ]
    )

    print(f"""{len(image_files)}""")
    results = []

    for img_path in tqdm(image_files, desc=""""""):
        try:
            result = predict_single(model, img_path, config, device, nms_distance=nms_distance)
            results.append(result)
        except Exception as e:
            print(f"""{img_path.name} : {e}""")

    # === ===
    num_keypoints = config["data"]["num_keypoints"]
    dot_counts = [r["num_dots"] for r in results]
    count_dist = Counter(dot_counts)

    total = len(results)
    full_yield = sum(1 for c in dot_counts if c == num_keypoints)  #
    partial = sum(1 for c in dot_counts if 0 < c < num_keypoints)  #
    missed = sum(1 for c in dot_counts if c == 0)  #
    excess = sum(1 for c in dot_counts if c > num_keypoints)  #

    yield_rate = full_yield / total * 100 if total > 0 else 0

    print(f"\n{'=' * 50}")
    print(f"""(: {num_keypoints} )""")
    print(f"{'=' * 50}")
    print(f""": {total}""")
    print(f"""( {num_keypoints} ): {full_yield} ( {full_yield / total * 100:.1f} %)""")
    print(f"""(0<dots< {num_keypoints} ): {partial} ( {partial / total * 100:.1f} %)""")
    print(f"""(0): {missed} ( {missed / total * 100:.1f} %)""")
    if excess > 0:
        print(f"""(> {num_keypoints} ): {excess} ( {excess / total * 100:.1f} %)""")
    print(f""": {yield_rate:.1f} %""")
    print(f"{'=' * 50}")

    print(f"""\n :""")
    for cnt in sorted(count_dist.keys()):
        c = count_dist[cnt]
        bar = "#" * int(c / total * 50) if total > 0 else ""
        print(f"""{cnt} : {c:4d} ( {c / total * 100:5.1f} %) {bar}""")

    # === ===
    if output_dir:
        if val_mode and checkpoint_path:
            checkpoint_path = Path(checkpoint_path)
            train_name = checkpoint_path.parent.parent.name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(output_dir) / f"val_{train_name}_{timestamp}"
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        #
        viz_dir = output_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)

        #
        for r in results:
            vis_img = visualize_yield(r, num_keypoints)
            cv2.imwrite(str(viz_dir / f"{r['image_name']}_pred.png"), vis_img)

        # JSON
        json_results = []
        for r in results:
            json_results.append(
                {
                    "image_name": r["image_name"],
                    "num_dots": r["num_dots"],
                    "expected_kp": r["expected_kp"],
                    "keypoints": r["keypoints"].tolist(),
                    "original_size": list(r["original_size"]),
                }
            )
        with open(output_dir / "results.json", "w", encoding="utf-8") as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)

        #
        yield_data = {
            "total": total,
            "expected_keypoints": num_keypoints,
            "full_detection": full_yield,
            "partial_detection": partial,
            "missed": missed,
            "excess": excess,
            "yield_rate": round(yield_rate, 2),
            "dot_distribution": {str(k): v for k, v in sorted(count_dist.items())},
            "model_config": {
                "hrnet_type": config["model"].get("hrnet_type", "N/A"),
                "checkpoint": str(checkpoint_path) if checkpoint_path else "N/A",
            },
        }
        with open(output_dir / "yield_report.json", "w", encoding="utf-8") as f:
            json.dump(yield_data, f, indent=2, ensure_ascii=False)

        #
        with open(output_dir / "yield_statistics.txt", "w", encoding="utf-8") as f:
            f.write(f"{'=' * 50}\n")
            f.write(f"""\n""")
            f.write(f""": {num_keypoints} \n""")
            f.write(f"{'=' * 50}\n\n")
            f.write(f""": {total} \n""")
            f.write(f"""( {num_keypoints} ): {full_yield} ( {full_yield / total * 100:.1f} %) \n""")
            f.write(f""": {partial} ( {partial / total * 100:.1f} %) \n""")
            f.write(f""": {missed} ( {missed / total * 100:.1f} %) \n""")
            if excess > 0:
                f.write(f""": {excess} ( {excess / total * 100:.1f} %) \n""")
            f.write(f""": {yield_rate:.1f} % \n \n""")
            f.write(""": \n""")
            for cnt in sorted(count_dist.keys()):
                f.write(f"""{cnt} : {count_dist[cnt]} \n""")

        print(f"""\n : {output_dir}""")

    return results


def visualize_yield(result, num_keypoints):
    """Args:
    result: predict_single()
    num_keypoints:"""
    img = cv2.imread(str(result["image_path"]), cv2.IMREAD_GRAYSCALE)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]

    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]
    keypoints = result["keypoints"]

    #
    for i, (x, y, conf) in enumerate(keypoints):
        if conf > 0:
            cx, cy = int(x), int(y)
            color = colors[i % len(colors)]
            cv2.circle(img, (cx, cy), 3, color, -1)
            cv2.circle(img, (cx, cy), 5, (255, 255, 255), 1)

    # : /
    num_dots = result["num_dots"]
    text = f"Dots: {num_dots}/{num_keypoints}"
    color = (
        (0, 255, 0)
        if num_dots == num_keypoints
        else (0, 0, 255)
        if num_dots == 0
        else (0, 255, 255)
    )
    cv2.putText(img, text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return img


def main():
    parser = argparse.ArgumentParser(description="""DNA -""")
    parser.add_argument("--type", choices=["0", "00", "01", "1", "10", "11"])
    parser.add_argument("--config", type=str)
    parser.add_argument("--weights", type=str, required=True, help="""""")
    parser.add_argument("--input", type=str, required=True, help="""""")
    parser.add_argument("--output", type=str, default="outputs/predictions", help="""""")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--val", action="store_true", help=""": outputs/val_results""")
    parser.add_argument("--nms-distance", type=int, default=None)
    args = parser.parse_args()

    if args.config is None:
        if args.type is None:
            parser.error("provide --type or --config")
        args.config = str(
            Path(__file__).resolve().parent / "configs" / "types" / f"type_{args.type}.yaml"
        )

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f""": {device}""")
    print(f""": {args.weights}""")
    print(f""": {config["data"]["num_keypoints"]}""")

    model = load_model(args.weights, config, device)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(args.input)

    if input_path.is_file():
        result = predict_single(model, input_path, config, device, nms_distance=args.nms_distance)
        print(f"""\n :""")
        print(f""": {result["image_name"]}""")
        print(f""": {result["num_dots"]} / {result["expected_kp"]}""")
        for i, (x, y, conf) in enumerate(result["keypoints"]):
            if conf > 0:
                print(f"""{i + 1} : ( {x:.1f} , {y:.1f} ), = {conf:.3f}""")

        if args.output:
            vis = visualize_yield(result, result["expected_kp"])
            viz_path = output_dir / f"{result['image_name']}_pred.png"
            cv2.imwrite(str(viz_path), vis)
            print(f""": {viz_path}""")

    elif input_path.is_dir():
        if args.val:
            output_dir = "outputs/val_results"
            print("""[]""")

        predict_batch(
            model,
            input_path,
            config,
            device,
            output_dir=output_dir,
            val_mode=args.val,
            checkpoint_path=args.weights,
            nms_distance=args.nms_distance,
        )
    else:
        print(f""": : {input_path}""")


if __name__ == "__main__":
    main()
