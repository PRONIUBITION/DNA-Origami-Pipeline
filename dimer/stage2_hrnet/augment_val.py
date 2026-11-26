"""Validation Set Data Augmentation Script
Only augment annotated images to generate variants such as rotations and flips."""

import os
import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil


def load_labelme_annotation(json_path):
    """Load Labelme annotations"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def save_labelme_annotation(data, output_path):
    """Save Labelme annotations"""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def has_annotation(data):
    """Check the annotation for actual keypoints (not [-1,-1])"""
    for shape in data.get("shapes", []):
        points = shape.get("points", [])
        if points and len(points) > 0:
            x, y = points[0]
            if x != -1 or y != -1:
                return True
    return False


def augment_image_and_keypoints(image, keypoints, aug_type="original"):
    """Enhance images and keypoints
    Return: (aug_image, aug_keypoints, success)"""
    h, w = image.shape[:2]
    aug_keypoints = keypoints.copy()

    if aug_type == "original":
        return image, aug_keypoints, True

    elif aug_type == "flip_lr":
        # Horizontal flip
        aug_image = cv2.flip(image, 1)
        # Synchronous flipping of keypoints
        for i in range(len(aug_keypoints)):
            if aug_keypoints[i][2] > 0:  # Visible points
                aug_keypoints[i][0] = w - aug_keypoints[i][0]
        return aug_image, aug_keypoints, True

    elif aug_type == "flip_ud":
        # Flip Vertically
        aug_image = cv2.flip(image, 0)
        # Synchronous flipping of keypoints
        for i in range(len(aug_keypoints)):
            if aug_keypoints[i][2] > 0:
                aug_keypoints[i][1] = h - aug_keypoints[i][1]
        return aug_image, aug_keypoints, True

    elif aug_type.startswith("rotate_"):
        # Rotate
        angle = int(aug_type.split("_")[1])
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        aug_image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        # Synchronous rotation of key points
        for i in range(len(aug_keypoints)):
            if aug_keypoints[i][2] > 0:
                x, y = aug_keypoints[i][0], aug_keypoints[i][1]
                # Apply rotation matrix
                new_x = M[0, 0] * x + M[0, 1] * y + M[0, 2]
                new_y = M[1, 0] * x + M[1, 1] * y + M[1, 2]
                aug_keypoints[i][0] = new_x
                aug_keypoints[i][1] = new_y
        return aug_image, aug_keypoints, True

    elif aug_type == "brightness_up":
        # Increase brightness
        aug_image = cv2.convertScaleAbs(image, alpha=1.0, beta=30)
        return aug_image, aug_keypoints, True

    elif aug_type == "brightness_down":
        # Decrease brightness
        aug_image = cv2.convertScaleAbs(image, alpha=1.0, beta=-30)
        return aug_image, aug_keypoints, True

    elif aug_type == "contrast_up":
        # Increase contrast
        aug_image = cv2.convertScaleAbs(image, alpha=1.2, beta=0)
        return aug_image, aug_keypoints, True

    elif aug_type == "contrast_down":
        # Decrease contrast
        aug_image = cv2.convertScaleAbs(image, alpha=0.8, beta=0)
        return aug_image, aug_keypoints, True

    return image, aug_keypoints, False


def update_annotation_with_keypoints(data, keypoints):
    """Update the annotation file with the new keypoint coordinates"""
    keypoint_order = {"left_dot_1": 0, "left_dot_2": 1, "right_dot_1": 2, "right_dot_2": 3}

    for shape in data.get("shapes", []):
        label = shape.get("label", "").lower()
        if label in keypoint_order:
            idx = keypoint_order[label]
            if keypoints[idx][2] > 0:  # Visible
                shape["points"] = [[float(keypoints[idx][0]), float(keypoints[idx][1])]]
            else:
                shape["points"] = [[-1, -1]]

    return data


def parse_keypoints_from_annotation(data, image_shape):
    """Parse keypoint coordinates from annotations"""
    h, w = image_shape[:2]
    keypoints = np.zeros((4, 3), dtype=np.float32)

    # Get the image size from the annotation
    anno_w = data.get("imageWidth", w)
    anno_h = data.get("imageHeight", h)
    scale_x = w / anno_w
    scale_y = h / anno_h

    keypoint_order = {"left_dot_1": 0, "left_dot_2": 1, "right_dot_1": 2, "right_dot_2": 3}

    for shape in data.get("shapes", []):
        label = shape.get("label", "").lower()
        if label in keypoint_order:
            idx = keypoint_order[label]
            points = shape.get("points", [])
            if points and len(points) > 0:
                x, y = points[0]
                if x == -1 and y == -1:
                    keypoints[idx] = [0, 0, 0]  # Invisible
                else:
                    keypoints[idx] = [x * scale_x, y * scale_y, 1]  # Visible

    return keypoints


def main():
    """Main function"""
    # Configuration path
    val_img_dir = Path("data/processed/val")
    val_anno_dir = Path("data/annotations/val")

    # Output path (overwrite directly or create a backup)
    output_img_dir = Path("data/processed/val_aug")
    output_anno_dir = Path("data/annotations/val_aug")

    # Clean up and create output directory
    if output_img_dir.exists():
        shutil.rmtree(output_img_dir)
    if output_anno_dir.exists():
        shutil.rmtree(output_anno_dir)

    output_img_dir.mkdir(parents=True, exist_ok=True)
    output_anno_dir.mkdir(parents=True, exist_ok=True)

    # Define enhancement type
    aug_types = [
        "original",
        "flip_lr",
        "flip_ud",
        "rotate_15",
        "rotate_-15",
        "brightness_up",
        "brightness_down",
    ]

    # Get all annotation files
    json_files = sorted(val_anno_dir.glob("*.json"))

    total_generated = 0
    skipped = 0

    print(f"""Starting to process the validation set augmentation...""")
    print(f"""{len(json_files)} annotation files found""")

    for json_file in tqdm(json_files, desc="""Processing annotation files"""):
        # Load annotation
        data = load_labelme_annotation(json_file)

        # Check for actual annotations
        if not has_annotation(data):
            skipped += 1
            continue

        # Get the corresponding image
        img_name = data.get("imagePath", json_file.stem + ".png")
        img_path = val_img_dir / img_name

        if not img_path.exists():
            # Try another extension
            for ext in [".png", ".jpg", ".jpeg"]:
                alt_path = val_img_dir / (json_file.stem + ext)
                if alt_path.exists():
                    img_path = alt_path
                    break

        if not img_path.exists():
            print(f"""Warning: Image not found {img_name}""")
            skipped += 1
            continue

        # Read image
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"""Warning: Unable to read image {img_path}""")
            skipped += 1
            continue

        # Parse keypoints
        keypoints = parse_keypoints_from_annotation(data, image.shape)

        # Generate various enhancements
        for aug_type in aug_types:
            try:
                aug_image, aug_keypoints, success = augment_image_and_keypoints(
                    image, keypoints, aug_type
                )

                if not success:
                    continue

                # Generate file name
                suffix = (
                    aug_type.replace("original", "aug0")
                    .replace("flip_lr", "aug1")
                    .replace("flip_ud", "aug2")
                    .replace("rotate_15", "aug3")
                    .replace("rotate_-15", "aug4")
                    .replace("brightness_up", "aug5")
                    .replace("brightness_down", "aug6")
                    .replace("contrast_up", "aug7")
                    .replace("contrast_down", "aug8")
                )
                new_name = f"{json_file.stem}_{suffix}"

                # Save Image
                output_img_path = output_img_dir / f"{new_name}.png"
                cv2.imwrite(str(output_img_path), aug_image)

                # Update and save annotation
                aug_data = data.copy()
                aug_data = update_annotation_with_keypoints(aug_data, aug_keypoints)
                aug_data["imagePath"] = f"{new_name}.png"
                aug_data["imageWidth"] = aug_image.shape[1]
                aug_data["imageHeight"] = aug_image.shape[0]

                output_anno_path = output_anno_dir / f"{new_name}.json"
                save_labelme_annotation(aug_data, output_anno_path)

                total_generated += 1

            except Exception as e:
                print(f"""Error processing {json_file.stem} of {aug_type}: {e}""")
                continue

    print(f"""\n Expansion complete!""")
    print(f"""Original annotated image: {len(json_files) - skipped}""")
    print(f"""Total number of images generated: {total_generated}""")
    print(f"""Variants generated per original image: {len(aug_types)}""")
    print(f"""Output Directory:""")
    print(f"""Image: {output_img_dir}""")
    print(f"""Annotation: {output_anno_dir}""")

    # List the number of files generated
    img_count = len(list(output_img_dir.glob("*.png")))
    anno_count = len(list(output_anno_dir.glob("*.json")))
    print(f"""\n Files actually generated:""")
    print(f"""Images: {img_count}""")
    print(f"""Annotations: {anno_count}""")


if __name__ == "__main__":
    main()
