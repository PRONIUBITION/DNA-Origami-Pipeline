"""Data Preparation Script
Prepare HRNet training data from the output of YOLOv5 Stage 1"""

import os
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm
import json
import random


def copy_from_yolo_output(yolo_output_dir, hrnet_raw_dir):
    """Copy the dimmer image from the YOLOv5 segmentation output to the HRNet data directory

    Args:
    yolo_output_dir: YOLOv5 clean_image.py output directory
    Example: "/path/to/masked-grayscale-images"
    hrnet_raw_dir: HRNet raw data directory"""
    yolo_output_dir = Path(yolo_output_dir)
    hrnet_raw_dir = Path(hrnet_raw_dir)

    if not yolo_output_dir.exists():
        raise ValueError(f"""YOLOv5 output directory does not exist: {yolo_output_dir}""")

    hrnet_raw_dir.mkdir(parents=True, exist_ok=True)

    # Find all clean images
    image_files = list(yolo_output_dir.glob("*_clean.png"))

    print(f"""{len(image_files)} images found in {yolo_output_dir}""")
    print(f"""Copy to {hrnet_raw_dir} ...""")

    copied = 0
    for img_path in tqdm(image_files):
        # Copy Image
        dest_path = hrnet_raw_dir / img_path.name
        shutil.copy2(img_path, dest_path)
        copied += 1

    print(f"""[OK] Successfully copied {copied} images""")
    return copied


def split_dataset(raw_dir, train_dir, val_dir, train_ratio=0.8, seed=42):
    """Divide into training set and validation set

    Args:
    raw_dir: Raw image directory
    train_dir: Training set directory
    val_dir: Validation set directory
    train_ratio: Training set ratio
    seed: Random seed"""
    raw_dir = Path(raw_dir)
    train_dir = Path(train_dir)
    val_dir = Path(val_dir)

    # Get all images
    image_files = sorted(
        [f for f in raw_dir.iterdir() if f.suffix.lower() in [".png", ".jpg", ".jpeg"]]
    )

    if len(image_files) == 0:
        print(f"""Warning: No images found in {raw_dir}""")
        return

    # Randomize
    random.seed(seed)
    random.shuffle(image_files)

    # Partition
    train_num = int(len(image_files) * train_ratio)
    train_files = image_files[:train_num]
    val_files = image_files[train_num:]

    # Create Directory
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    # Copy training set
    print(f"""\n Partition the dataset:""")
    print(f"""Training set: {len(train_files)} images ( {train_ratio * 100:.0f} %)""")
    print(f"""Validation set: {len(val_files)} images ( {(1 - train_ratio) * 100:.0f} %)""")

    for f in tqdm(train_files, desc="""Copy training set"""):
        shutil.copy2(f, train_dir / f.name)

    for f in tqdm(val_files, desc="""Copy validation set"""):
        shutil.copy2(f, val_dir / f.name)

    print("""[OK] Dataset partitioning completed""")


def create_labelme_template(image_dir, output_dir):
    """Create a Labelme annotation template (empty annotation file) for each image

    Args:
    image_dir: Image directory
    output_dir: Annotation output directory"""
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(
        [f for f in image_dir.iterdir() if f.suffix.lower() in [".png", ".jpg", ".jpeg"]]
    )

    print(f"""\n Creating Labelme annotation template...""")

    for img_path in tqdm(image_files):
        # Read image dimensions
        from PIL import Image

        with Image.open(img_path) as img:
            width, height = img.size

        # Create an empty Labelme annotation file
        # Note: If the imageData field is not added, Labelme will automatically load the image from imagePath.
        annotation = {
            "version": "5.0.1",
            "flags": {},
            "shapes": [],  # Empty annotation; small white dots need to be added manually.
            "imagePath": str(img_path.name),
            "imageHeight": height,
            "imageWidth": width,
        }

        # Save
        json_path = output_dir / f"{img_path.stem}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(annotation, f, indent=2, ensure_ascii=False)

    print(f"""[OK] {len(image_files)} annotation templates created""")
    print(f"""Saved in: {output_dir}""")


def check_annotations(image_dir, anno_dir):
    """Check if the annotation is complete

    Args:
    image_dir: Image directory
    anno_dir: Annotation directory"""
    image_dir = Path(image_dir)
    anno_dir = Path(anno_dir)

    image_files = sorted(
        [f.stem for f in image_dir.iterdir() if f.suffix.lower() in [".png", ".jpg", ".jpeg"]]
    )

    anno_files = sorted([f.stem for f in anno_dir.iterdir() if f.suffix == ".json"])

    missing = set(image_files) - set(anno_files)
    extra = set(anno_files) - set(image_files)

    print("""\n Annotation check:""")
    print(f"""Number of images: {len(image_files)}""")
    print(f"""Number of annotations: {len(anno_files)}""")

    if missing:
        print(f"""Missing annotations: {len(missing)}""")
        for name in list(missing)[:5]:
            print(f"    - {name}")
    else:
        print(f"""[OK] All images have annotations""")

    if extra:
        print(f"""Excess annotations: {len(extra)}""")

    # Count the distribution of each category (number of white dots)
    dot_counts = []
    for anno_file in anno_dir.glob("*.json"):
        with open(anno_file, "r") as f:
            data = json.load(f)
        num_dots = len([s for s in data.get("shapes", []) if "dot" in s.get("label", "").lower()])
        dot_counts.append(num_dots)

    from collections import Counter

    distribution = Counter(dot_counts)

    print(f"""\n Distribution of the number of white dots:""")
    for num_dots in sorted(distribution.keys()):
        count = distribution[num_dots]
        print(f"""{num_dots} points: {count} images ( {count / len(dot_counts) * 100:.1f} %)""")


def main():
    parser = argparse.ArgumentParser(description="Prepare data for HRNet training")
    parser.add_argument(
        "--yolo_output", type=str, default=None, help="""YOLOv5 Stage 1 Output Directory Path"""
    )
    parser.add_argument(
        "--split", action="store_true", help="""Divide the training set and the validation set"""
    )
    parser.add_argument(
        "--create_template", action="store_true", help="""Create a Labelme annotation template"""
    )
    parser.add_argument("--check", action="store_true", help="""Check annotation completeness""")
    args = parser.parse_args()

    # Project root directory
    project_root = Path(__file__).parent
    data_dir = project_root / "data"

    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    anno_dir = data_dir / "annotations"

    # Step 1: Copy data from YOLOv5
    if args.yolo_output:
        print("=" * 50)
        print("""Step 1: Copy data from YOLOv5""")
        print("=" * 50)

        yolo_path = Path(args.yolo_output)
        if not yolo_path.is_absolute():
            # Relative path, based on the project root directory
            yolo_path = project_root.parent / args.yolo_output

        copy_from_yolo_output(yolo_path, raw_dir)

    # Step 2: Divide the dataset
    if args.split:
        print("\n" + "=" * 50)
        print("""Step 2: Divide the training set and the validation set""")
        print("=" * 50)

        train_dir = processed_dir / "train"
        val_dir = processed_dir / "val"

        split_dataset(raw_dir, train_dir, val_dir, train_ratio=0.8)

    # Step 3: Create an annotation template
    if args.create_template:
        print("\n" + "=" * 50)
        print("""Step 3: Create a Labelme annotation template""")
        print("=" * 50)

        # Create templates for both the training set and the validation set
        if (processed_dir / "train").exists():
            create_labelme_template(processed_dir / "train", anno_dir / "train")

        if (processed_dir / "val").exists():
            create_labelme_template(processed_dir / "val", anno_dir / "val")

        print("""\n [INFO] Annotation instructions:""")
        print("""1. Open the Labelme software""")
        print("""2. Open the annotation directory: {}""".format(anno_dir / "train"))
        print("""3. For each image, use the "Point" tool to annotate the small white dots.""")
        print("""4. Name the labels: dot_1, dot_2, dot_3, dot_4 (in order)""")
        print("""5. Save the annotation""")

    # Step 4: Check the annotation
    if args.check:
        print("\n" + "=" * 50)
        print("""Step 4: Check the completeness of the annotation""")
        print("=" * 50)

        if (processed_dir / "train").exists():
            print("""\n Training set:""")
            check_annotations(processed_dir / "train", anno_dir / "train")

        if (processed_dir / "val").exists():
            print("""\n Validation set:""")
            check_annotations(processed_dir / "val", anno_dir / "val")

    print("\n" + "=" * 50)
    print("""Data preparation complete!""")
    print("=" * 50)
    print(f"""Data Directory: {data_dir}""")
    print(f"""- Original image: {raw_dir}""")
    print(f"""- Processed image: {processed_dir}""")
    print(f"""- Annotation file: {anno_dir}""")


if __name__ == "__main__":
    main()
