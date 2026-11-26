import json
import os
import yaml
import shutil
import random
from pathlib import Path
from settings import DATA_ROOT, OUTPUT_ROOT


def convert_labelme_to_yolo_seg(json_dir, output_dir, img_width, img_height, class_names):
    """Labelme YOLO-Seg
    , bbox"""
    os.makedirs(output_dir, exist_ok=True)
    class_to_id = {name: idx for idx, name in enumerate(class_names)}

    json_files = list(Path(json_dir).glob("*.json"))
    print(f"""{len(json_files)} JSON""")

    for json_path in json_files:
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            #
            image_path = data.get("imagePath", "")
            base_name = Path(image_path).stem or json_path.stem
            txt_path = Path(output_dir) / f"{base_name}.txt"

            lines = []

            for shape in data.get("shapes", []):
                label = shape.get("label", "").strip()

                #
                if label not in class_to_id:
                    print(f"""[WARN] ' {label} ' in {json_path.name}""")
                    continue

                points = shape.get("points", [])

                # 3
                if len(points) < 3:
                    print(f"""[WARN] ( {len(points)} ) in {json_path.name}""")
                    continue

                #
                normalized_points = []
                for x, y in points:
                    x_norm = float(x) / img_width
                    y_norm = float(y) / img_height
                    # [0, 1]
                    x_norm = max(0.0, min(1.0, x_norm))
                    y_norm = max(0.0, min(1.0, y_norm))
                    normalized_points.extend([f"{x_norm:.6f}", f"{y_norm:.6f}"])

                # YOLO-Seg : <class> <x1> <y1> <x2> <y2> ...
                line = f"{class_to_id[label]} " + " ".join(normalized_points)
                lines.append(line)

            #
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))

            print(
                f"""[OK] {json_path.name} -> {txt_path.name} ( {len(lines)} , {sum(len(l.split()) - 1 for l in lines) // 2} )"""
            )

        except Exception as e:
            print(f"""[FAIL] {json_path.name} : {e}""")


def split_and_convert_dataset(config):
    """: + + yaml"""

    # 1. ()
    raw_image_dir = Path(config["raw_image_dir"])
    raw_json_dir = Path(config["raw_json_dir"])
    dataset_dir = Path(config["dataset_dir"])

    #
    image_files = list(raw_image_dir.glob("*.jpg")) + list(raw_image_dir.glob("*.png"))
    image_files.sort()
    random.seed(42)
    random.shuffle(image_files)

    total = len(image_files)
    if total == 0:
        raise ValueError("""!""")

    print(f""": {total}""")

    #
    train_num = int(total * config["train_ratio"])
    train_files = image_files[:train_num]
    val_files = image_files[train_num:]

    print(f""": {len(train_files)} ( {len(train_files) / total * 100:.1f} %)""")
    print(f""": {len(val_files)} ( {len(val_files) / total * 100:.1f} %)""")

    #
    for split_name, files in [("train", train_files), ("val", val_files)]:
        (dataset_dir / "images" / split_name).mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / split_name).mkdir(parents=True, exist_ok=True)

        for img_path in files:
            #
            shutil.copy(img_path, dataset_dir / "images" / split_name / img_path.name)

            # JSON
            json_path = raw_json_dir / (img_path.stem + ".json")
            if json_path.exists():
                json_dest = dataset_dir / "jsons" / split_name
                json_dest.mkdir(parents=True, exist_ok=True)
                shutil.copy(json_path, json_dest / json_path.name)

    # 2. ()
    print(f"""\n (YOLO-Seg)...""")
    convert_labelme_to_yolo_seg(
        json_dir=dataset_dir / "jsons" / "train",
        output_dir=dataset_dir / "labels" / "train",
        img_width=config["img_width"],
        img_height=config["img_height"],
        class_names=config["class_names"],
    )

    # 3.
    print(f"""\n (YOLO-Seg)...""")
    convert_labelme_to_yolo_seg(
        json_dir=dataset_dir / "jsons" / "val",
        output_dir=dataset_dir / "labels" / "val",
        img_width=config["img_width"],
        img_height=config["img_height"],
        class_names=config["class_names"],
    )

    # 4. data.yaml
    yaml_data = {
        "path": str(dataset_dir).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "nc": len(config["class_names"]),
        "names": {i: name for i, name in enumerate(config["class_names"])},
    }

    with open(config["yaml_path"], "w", encoding="utf-8") as f:
        yaml.dump(yaml_data, f, sort_keys=False, allow_unicode=True, default_flow_style=False)

    print(f"""\n [OK] : {config["yaml_path"]}""")


# ===== =====
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Labelme -> YOLO-Seg")
    parser.add_argument("--type", required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--class-name", default="monomer")
    args = parser.parse_args()
    base_dir = DATA_ROOT

    CONFIG = {
        "raw_image_dir": str(base_dir / "yolo_datasets" / args.type / "images" / "cropped"),
        "raw_json_dir": str(base_dir / "yolo_datasets" / args.type / "labels" / "json"),
        "dataset_dir": str(OUTPUT_ROOT / "yolo_datasets" / args.type / "yolo_data"),
        "yaml_path": str(OUTPUT_ROOT / "yolo_datasets" / args.type / "data.yaml"),
        "img_width": args.width,
        "img_height": args.height,
        "class_names": [args.class_name],
        "train_ratio": 0.8,
    }
    print("""...\ \n""")
    split_and_convert_dataset(CONFIG)
    print("""!""")
