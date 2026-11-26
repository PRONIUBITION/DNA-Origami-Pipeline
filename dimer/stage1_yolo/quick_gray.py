import json
import os
import yaml
import shutil
import random
import cv2
from pathlib import Path
from settings import data_path, output_path


def convert_images_to_grayscale(input_dir, output_dir):
    """Convert image to grayscale and save"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    converted = []

    for img_path in Path(input_dir).glob("*"):
        if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]:
            # Read as grayscale
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                # Save as 3-channel grayscale (YOLO-compatible) or single-channel
                output_path = Path(output_dir) / (img_path.stem + ".png")
                cv2.imwrite(str(output_path), img)
                converted.append(img_path.stem)
                print(f"""{img_path.name} -> Grayscale""")

    return converted


def convert_labelme_to_yolo_seg(json_dir, output_dir, img_width, img_height, class_names):
    """YOLO-Seg format conversion (unchanged)"""
    os.makedirs(output_dir, exist_ok=True)
    class_to_id = {name: idx for idx, name in enumerate(class_names)}

    for json_path in Path(json_dir).glob("*.json"):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        base_name = Path(data.get("imagePath", json_path.stem)).stem
        txt_path = Path(output_dir) / f"{base_name}.txt"

        lines = []
        for shape in data.get("shapes", []):
            label = shape.get("label", "").strip()
            if label not in class_to_id:
                continue

            points = shape.get("points", [])
            if len(points) < 3:
                continue

            normalized = []
            for x, y in points:
                x_norm = max(0.0, min(1.0, float(x) / img_width))
                y_norm = max(0.0, min(1.0, float(y) / img_height))
                normalized.extend([f"{x_norm:.6f}", f"{y_norm:.6f}"])

            lines.append(f"{class_to_id[label]} " + " ".join(normalized))

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))


# ===== Main Process =====
if __name__ == "__main__":
    CONFIG = {
        "raw_image_dir": data_path("annotations", "images"),
        "raw_json_dir": data_path("annotations", "json"),
        "dataset_dir": output_path("yolo_gray_dataset"),
        "yaml_path": output_path("yolo_gray_dataset", "data.yaml"),
        "img_width": 516,
        "img_height": 514,
        "class_names": ["dimmer", "monomer", "invalid"],
        "train_ratio": 0.8,
    }

    print("""Generating a grayscale version of the YOLO-Seg dataset... \n""")

    # 1. Segment and convert images to grayscale
    raw_images = list(Path(CONFIG["raw_image_dir"]).glob("*"))
    random.seed(42)
    random.shuffle(raw_images)

    train_num = int(len(raw_images) * CONFIG["train_ratio"])
    train_files = raw_images[:train_num]
    val_files = raw_images[train_num:]

    print(
        f"""Total samples: {len(raw_images)}, Training: {len(train_files)}, Validation: {len(val_files)}"""
    )

    # Process training set
    (Path(CONFIG["dataset_dir"]) / "images" / "train").mkdir(parents=True, exist_ok=True)
    (Path(CONFIG["dataset_dir"]) / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (Path(CONFIG["dataset_dir"]) / "jsons" / "train").mkdir(parents=True, exist_ok=True)

    for img_path in train_files:
        # Convert to grayscale
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            cv2.imwrite(
                str(Path(CONFIG["dataset_dir"]) / "images" / "train" / (img_path.stem + ".png")),
                img,
            )
        # Copy JSON
        json_path = Path(CONFIG["raw_json_dir"]) / (img_path.stem + ".json")
        if json_path.exists():
            shutil.copy(json_path, Path(CONFIG["dataset_dir"]) / "jsons" / "train" / json_path.name)

    # Process the validation set (similarly)
    (Path(CONFIG["dataset_dir"]) / "images" / "val").mkdir(parents=True, exist_ok=True)
    (Path(CONFIG["dataset_dir"]) / "labels" / "val").mkdir(parents=True, exist_ok=True)
    (Path(CONFIG["dataset_dir"]) / "jsons" / "val").mkdir(parents=True, exist_ok=True)

    for img_path in val_files:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            cv2.imwrite(
                str(Path(CONFIG["dataset_dir"]) / "images" / "val" / (img_path.stem + ".png")), img
            )
        json_path = Path(CONFIG["raw_json_dir"]) / (img_path.stem + ".json")
        if json_path.exists():
            shutil.copy(json_path, Path(CONFIG["dataset_dir"]) / "jsons" / "val" / json_path.name)

    # 2. Convert labels
    print("""\n Converting training labels...""")
    convert_labelme_to_yolo_seg(
        Path(CONFIG["dataset_dir"]) / "jsons" / "train",
        Path(CONFIG["dataset_dir"]) / "labels" / "train",
        CONFIG["img_width"],
        CONFIG["img_height"],
        CONFIG["class_names"],
    )

    print("""\n Converting validation labels...""")
    convert_labelme_to_yolo_seg(
        Path(CONFIG["dataset_dir"]) / "jsons" / "val",
        Path(CONFIG["dataset_dir"]) / "labels" / "val",
        CONFIG["img_width"],
        CONFIG["img_height"],
        CONFIG["class_names"],
    )

    # 3. Generate YAML
    yaml_data = {
        "path": str(CONFIG["dataset_dir"]).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "nc": 2,
        "names": {0: "dimmer", 1: "fragment"},
    }
    with open(CONFIG["yaml_path"], "w") as f:
        yaml.dump(yaml_data, f, sort_keys=False, allow_unicode=True)

    print(f"""\n Done! Gray-scale dataset: {CONFIG["dataset_dir"]}""")
    print(f"""Training command:""")
    print(
        f'python segment/train.py --data "{CONFIG["yaml_path"]}" --weights yolov5n-seg.pt --img 320 --batch 16 --epochs 50'
    )
