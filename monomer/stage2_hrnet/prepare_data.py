"""YOLOv51HRNet"""

import os
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm
import json
import random


def copy_from_yolo_output(yolo_output_dir, hrnet_raw_dir):
    """YOLOv5dimmerHRNet

    Args:
    yolo_output_dir: YOLOv5 clean_image.py
    : "/path/to/masked-grayscale-images"
    hrnet_raw_dir: HRNet"""
    yolo_output_dir = Path(yolo_output_dir)
    hrnet_raw_dir = Path(hrnet_raw_dir)

    if not yolo_output_dir.exists():
        raise ValueError(f"""YOLOv5: {yolo_output_dir}""")

    hrnet_raw_dir.mkdir(parents=True, exist_ok=True)

    #
    image_files = list(yolo_output_dir.glob("*_clean.png"))

    print(f"""{yolo_output_dir} {len(image_files)}""")
    print(f"""{hrnet_raw_dir} ...""")

    copied = 0
    for img_path in tqdm(image_files):
        #
        dest_path = hrnet_raw_dir / img_path.name
        shutil.copy2(img_path, dest_path)
        copied += 1

    print(f"""[OK] {copied}""")
    return copied


def split_dataset(raw_dir, train_dir, val_dir, train_ratio=0.8, seed=42):
    """Args:
    raw_dir:
    train_dir:
    val_dir:
    train_ratio:
    seed:"""
    raw_dir = Path(raw_dir)
    train_dir = Path(train_dir)
    val_dir = Path(val_dir)

    #
    image_files = sorted(
        [f for f in raw_dir.iterdir() if f.suffix.lower() in [".png", ".jpg", ".jpeg"]]
    )

    if len(image_files) == 0:
        print(f""": {raw_dir}""")
        return

    #
    random.seed(seed)
    random.shuffle(image_files)

    #
    train_num = int(len(image_files) * train_ratio)
    train_files = image_files[:train_num]
    val_files = image_files[train_num:]

    #
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    #
    print(f"""\n :""")
    print(f""": {len(train_files)} ( {train_ratio * 100:.0f} %)""")
    print(f""": {len(val_files)} ( {(1 - train_ratio) * 100:.0f} %)""")

    for f in tqdm(train_files, desc=""""""):
        shutil.copy2(f, train_dir / f.name)

    for f in tqdm(val_files, desc=""""""):
        shutil.copy2(f, val_dir / f.name)

    print("""[OK]""")


def create_labelme_template(image_dir, output_dir):
    """Labelme()

    Args:
    image_dir:
    output_dir:"""
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(
        [f for f in image_dir.iterdir() if f.suffix.lower() in [".png", ".jpg", ".jpeg"]]
    )

    print(f"""\n Labelme...""")

    for img_path in tqdm(image_files):
        #
        from PIL import Image

        with Image.open(img_path) as img:
            width, height = img.size

        # Labelme
        # :imageData,LabelmeimagePath
        annotation = {
            "version": "5.0.1",
            "flags": {},
            "shapes": [],  # ,
            "imagePath": str(img_path.name),
            "imageHeight": height,
            "imageWidth": width,
        }

        #
        json_path = output_dir / f"{img_path.stem}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(annotation, f, indent=2, ensure_ascii=False)

    print(f"""[OK] {len(image_files)}""")
    print(f""": {output_dir}""")


def check_annotations(image_dir, anno_dir):
    """Args:
    image_dir:
    anno_dir:"""
    image_dir = Path(image_dir)
    anno_dir = Path(anno_dir)

    image_files = sorted(
        [f.stem for f in image_dir.iterdir() if f.suffix.lower() in [".png", ".jpg", ".jpeg"]]
    )

    anno_files = sorted([f.stem for f in anno_dir.iterdir() if f.suffix == ".json"])

    missing = set(image_files) - set(anno_files)
    extra = set(anno_files) - set(image_files)

    print("""\n :""")
    print(f""": {len(image_files)}""")
    print(f""": {len(anno_files)}""")

    if missing:
        print(f""": {len(missing)}""")
        for name in list(missing)[:5]:
            print(f"    - {name}")
    else:
        print(f"""[OK]""")

    if extra:
        print(f""": {len(extra)}""")

    # ()
    dot_counts = []
    for anno_file in anno_dir.glob("*.json"):
        with open(anno_file, "r") as f:
            data = json.load(f)
        num_dots = len([s for s in data.get("shapes", []) if "dot" in s.get("label", "").lower()])
        dot_counts.append(num_dots)

    from collections import Counter

    distribution = Counter(dot_counts)

    print(f"""\n :""")
    for num_dots in sorted(distribution.keys()):
        count = distribution[num_dots]
        print(f"""{num_dots} : {count} ( {count / len(dot_counts) * 100:.1f} %)""")


def main():
    parser = argparse.ArgumentParser(description="Prepare data for HRNet training")
    parser.add_argument("--yolo_output", type=str, default=None, help="""YOLOv51""")
    parser.add_argument("--split", action="store_true", help="""""")
    parser.add_argument("--create_template", action="store_true", help="""Labelme""")
    parser.add_argument("--check", action="store_true", help="""""")
    args = parser.parse_args()

    #
    project_root = Path(__file__).parent
    data_dir = project_root / "data"

    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    anno_dir = data_dir / "annotations"

    # 1: YOLOv5
    if args.yolo_output:
        print("=" * 50)
        print("""1: YOLOv5""")
        print("=" * 50)

        yolo_path = Path(args.yolo_output)
        if not yolo_path.is_absolute():
            # ,
            yolo_path = project_root.parent / args.yolo_output

        copy_from_yolo_output(yolo_path, raw_dir)

    # 2:
    if args.split:
        print("\n" + "=" * 50)
        print("""2:""")
        print("=" * 50)

        train_dir = processed_dir / "train"
        val_dir = processed_dir / "val"

        split_dataset(raw_dir, train_dir, val_dir, train_ratio=0.8)

    # 3:
    if args.create_template:
        print("\n" + "=" * 50)
        print("""3: Labelme""")
        print("=" * 50)

        #
        if (processed_dir / "train").exists():
            create_labelme_template(processed_dir / "train", anno_dir / "train")

        if (processed_dir / "val").exists():
            create_labelme_template(processed_dir / "val", anno_dir / "val")

        print("""\n [INFO] :""")
        print("""1. Labelme""")
        print("""2. : {}""".format(anno_dir / "train"))
        print("""3. ,''""")
        print("""4. : dot_1, dot_2, dot_3, dot_4 ()""")
        print("""5.""")

    # 4:
    if args.check:
        print("\n" + "=" * 50)
        print("""4:""")
        print("=" * 50)

        if (processed_dir / "train").exists():
            print("""\n :""")
            check_annotations(processed_dir / "train", anno_dir / "train")

        if (processed_dir / "val").exists():
            print("""\n :""")
            check_annotations(processed_dir / "val", anno_dir / "val")

    print("\n" + "=" * 50)
    print("""!""")
    print("=" * 50)
    print(f""": {data_dir}""")
    print(f"""- : {raw_dir}""")
    print(f"""- : {processed_dir}""")
    print(f"""- : {anno_dir}""")


if __name__ == "__main__":
    main()
