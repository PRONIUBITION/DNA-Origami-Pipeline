import json
import os
import yaml
import shutil
import random
from pathlib import Path
from settings import data_path, output_path


def convert_labelme_to_yolo_seg(json_dir, output_dir, img_width, img_height, class_names):
    """Convert Labelme Polygon Annotations to YOLO-Seg Format
    Retain the complete polygon outline without converting it to a bounding box"""
    os.makedirs(output_dir, exist_ok=True)
    class_to_id = {name: idx for idx, name in enumerate(class_names)}

    json_files = list(Path(json_dir).glob("*.json"))
    print(f"""{len(json_files)} JSON files found""")

    for json_path in json_files:
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Get image name
            image_path = data.get("imagePath", "")
            base_name = Path(image_path).stem or json_path.stem
            txt_path = Path(output_dir) / f"{base_name}.txt"

            lines = []

            for shape in data.get("shapes", []):
                label = shape.get("label", "").strip()

                # Skip undefined categories
                if label not in class_to_id:
                    print(f"""Skip unknown category ' {label} ' in {json_path.name}""")
                    continue

                points = shape.get("points", [])

                # Polygons require at least 3 points
                if len(points) < 3:
                    print(
                        f"""Skip polygons with insufficient points ( {len(points)} points) in {json_path.name}"""
                    )
                    continue

                # Normalize all point coordinates
                normalized_points = []
                for x, y in points:
                    x_norm = float(x) / img_width
                    y_norm = float(y) / img_height
                    # Crop to the range [0, 1]
                    x_norm = max(0.0, min(1.0, x_norm))
                    y_norm = max(0.0, min(1.0, y_norm))
                    normalized_points.extend([f"{x_norm:.6f}", f"{y_norm:.6f}"])

                # Building YOLO-Seg row:<class> <x1> <y1> <x2> <y2>...
                line = f"{class_to_id[label]} " + " ".join(normalized_points)
                lines.append(line)

            # Write to file
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))

            print(
                f"""{json_path.name} -> {txt_path.name} ( {len(lines)} polygons, {sum(len(l.split()) - 1 for l in lines) // 2} total vertices)"""
            )

        except Exception as e:
            print(f"""Failed {json_path.name} : {e}""")


def split_and_convert_dataset(config):
    """One-stop processing: Dataset splitting + Label conversion + YAML generation"""

    # 1. Divide the dataset (if not already divided)
    raw_image_dir = Path(config["raw_image_dir"])
    raw_json_dir = Path(config["raw_json_dir"])
    dataset_dir = Path(config["dataset_dir"])

    # Get all images
    image_files = list(raw_image_dir.glob("*.jpg")) + list(raw_image_dir.glob("*.png"))
    image_files.sort()
    random.seed(42)
    random.shuffle(image_files)

    total = len(image_files)
    if total == 0:
        raise ValueError("""No image file found!""")

    print(f"""Total number of samples: {total}""")

    # Partition ratio
    train_num = int(total * config["train_ratio"])
    train_files = image_files[:train_num]
    val_files = image_files[train_num:]

    print(f"""Training set: {len(train_files)} ( {len(train_files) / total * 100:.1f} %)""")
    print(f"""Validation set: {len(val_files)} ( {len(val_files) / total * 100:.1f} %)""")

    # Create directory structure
    for split_name, files in [("train", train_files), ("val", val_files)]:
        (dataset_dir / "images" / split_name).mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / split_name).mkdir(parents=True, exist_ok=True)

        for img_path in files:
            # Copy image
            shutil.copy(img_path, dataset_dir / "images" / split_name / img_path.name)

            # Copy the corresponding JSON
            json_path = raw_json_dir / (img_path.stem + ".json")
            if json_path.exists():
                json_dest = dataset_dir / "jsons" / split_name
                json_dest.mkdir(parents=True, exist_ok=True)
                shutil.copy(json_path, json_dest / json_path.name)

    # 2. Convert training set labels (polygon format)
    print(f"""\n Converting training set labels (YOLO-Seg format)...""")
    convert_labelme_to_yolo_seg(
        json_dir=dataset_dir / "jsons" / "train",
        output_dir=dataset_dir / "labels" / "train",
        img_width=config["img_width"],
        img_height=config["img_height"],
        class_names=config["class_names"],
    )

    # 3. Convert validation set labels
    print(f"""\n Converting validation set labels (YOLO-Seg format)...""")
    convert_labelme_to_yolo_seg(
        json_dir=dataset_dir / "jsons" / "val",
        output_dir=dataset_dir / "labels" / "val",
        img_width=config["img_width"],
        img_height=config["img_height"],
        class_names=config["class_names"],
    )

    # 4. Generate data.yaml
    yaml_data = {
        "path": str(dataset_dir).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "nc": len(config["class_names"]),
        "names": {i: name for i, name in enumerate(config["class_names"])},
    }

    with open(config["yaml_path"], "w", encoding="utf-8") as f:
        yaml.dump(yaml_data, f, sort_keys=False, allow_unicode=True, default_flow_style=False)

    print(f"""\n Configuration file: {config["yaml_path"]}""")


# ===== Configuration and Operation =====
if __name__ == "__main__":
    CONFIG = {
        "raw_image_dir": data_path("annotations", "images"),
        "raw_json_dir": data_path("annotations", "json"),
        "dataset_dir": output_path("yolo_dataset"),
        "yaml_path": output_path("yolo_dataset", "data.yaml"),
        "img_width": 516,  # Your image width
        "img_height": 514,  # Your image height
        "class_names": ["dimmer", "monomer", "invalid"],
        "train_ratio": 0.8,  # 80% training, 20% validation
    }

    print("""Starting to process the polygon segmentation dataset... \n""")
    split_and_convert_dataset(CONFIG)

    print("\n" + "=" * 60)
    print("""Done! The YOLO-Seg format dataset is ready.""")
    print("=" * 60)
    print(f"""\n Training command (note the use of segment/train.py):""")
    print(
        f'python segment/train.py --data "{CONFIG["yaml_path"]}" --weights yolov5s-seg.pt --img 516 --batch 8 --epochs 100 --name dna_seg_v1'
    )
