"""-
PIL + OpenCV, OpenCV"""

import os
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageEnhance, ImageFilter
import random
import cv2


def load_labelme_annotation(json_path):
    """Labelme"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    points = []
    labels = []
    for shape in data.get("shapes", []):
        if shape.get("shape_type") == "point":
            points.append(shape["points"][0])  # [x, y]
            labels.append(shape.get("label", "dot"))

    return {
        "points": np.array(points) if points else np.array([]),
        "labels": labels,
        "imagePath": data.get("imagePath", ""),
        "imageHeight": data.get("imageHeight", 0),
        "imageWidth": data.get("imageWidth", 0),
    }


def save_labelme_annotation(json_path, points, labels, image_path, height, width):
    """Labelme"""
    shapes = []
    for pt, label in zip(points, labels):
        shapes.append(
            {
                "label": label,
                "points": [[float(pt[0]), float(pt[1])]],
                "shape_type": "point",
                "flags": {},
            }
        )

    annotation = {
        "version": "5.0.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": str(image_path),
        "imageHeight": int(height),
        "imageWidth": int(width),
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(annotation, f, indent=2, ensure_ascii=False)


def rotate_image_and_points(image, points, angle):
    """angle: 90, 180, 270"""
    if angle == 0:
        return image, points

    w, h = image.size

    #
    rotated = image.rotate(angle, expand=False, fillcolor=0)

    #
    if len(points) > 0:
        angle_rad = np.radians(angle)
        cx, cy = w / 2, h / 2

        rotated_points = points.copy()
        for i, (x, y) in enumerate(points):
            #
            x_shifted = x - cx
            y_shifted = y - cy
            #
            x_rot = x_shifted * np.cos(angle_rad) - y_shifted * np.sin(angle_rad)
            y_rot = x_shifted * np.sin(angle_rad) + y_shifted * np.cos(angle_rad)
            #
            rotated_points[i] = [x_rot + cx, y_rot + cy]
    else:
        rotated_points = points

    return rotated, rotated_points


def flip_image_and_points(image, points, flip_type):
    """flip_type: 'h' , 'v'"""
    w, h = image.size

    if flip_type == "h":
        flipped = image.transpose(Image.FLIP_LEFT_RIGHT)
        if len(points) > 0:
            flipped_points = points.copy()
            flipped_points[:, 0] = w - points[:, 0]
        else:
            flipped_points = points
    else:  # vertical
        flipped = image.transpose(Image.FLIP_TOP_BOTTOM)
        if len(points) > 0:
            flipped_points = points.copy()
            flipped_points[:, 1] = h - points[:, 1]
        else:
            flipped_points = points

    return flipped, flipped_points


def elastic_transform(image, points, alpha=20, sigma=4):
    import cv2

    w, h = image.size
    img_array = np.array(image, dtype=np.float32)
    dx = np.random.RandomState(None).rand(h, w) * 2 - 1
    dy = np.random.RandomState(None).rand(h, w) * 2 - 1
    k = max(3, int(sigma * 3) | 1)
    dx = cv2.GaussianBlur(dx, (k, k), sigma) * alpha
    dy = cv2.GaussianBlur(dy, (k, k), sigma) * alpha
    x_grid, y_grid = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (x_grid + dx).astype(np.float32)
    map_y = (y_grid + dy).astype(np.float32)
    transformed = cv2.remap(
        img_array, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )
    new_points = points.copy() if len(points) > 0 else points
    if len(points) > 0:
        for i, (px, py) in enumerate(points):
            ix = np.clip(int(round(px)), 0, w - 1)
            iy = np.clip(int(round(py)), 0, h - 1)
            new_points[i, 0] = px + dx[iy, ix]
            new_points[i, 1] = py + dy[iy, ix]
    return Image.fromarray(transformed.astype(np.uint8)), new_points


def cutout(image, points, num_holes=2, max_hole_size=40):
    w, h = image.size
    arr = np.array(image)
    for _ in range(num_holes):
        hw = random.randint(10, max_hole_size)
        hh = random.randint(10, max_hole_size)
        x = random.randint(0, max(1, w - hw))
        y = random.randint(0, max(1, h - hh))
        arr[y : y + hh, x : x + hw] = 0
    return Image.fromarray(arr), points


def add_gaussian_noise(image, points, std=15):
    arr = np.array(image)
    n = np.random.normal(0, std, arr.shape)
    return Image.fromarray(np.clip(arr.astype(np.float32) + n, 0, 255).astype(np.uint8)), points


def gaussian_blur(image, points, radius=1.5):
    return image.filter(ImageFilter.GaussianBlur(radius=radius)), points


def sharpen_image(image, points):
    return image.filter(ImageFilter.SHARPEN), points


def adjust_contrast(image, points, factor):
    return ImageEnhance.Contrast(image).enhance(factor), points


def rotate_small(image, points, angle):
    w, h = image.size
    r = image.rotate(angle, expand=False, fillcolor=0, resample=Image.BICUBIC)
    if len(points) == 0:
        return r, points
    cx, cy = w / 2, h / 2
    ar = np.radians(angle)
    ca, sa = np.cos(ar), np.sin(ar)
    new_pts = points.copy()
    for i, (x, y) in enumerate(points):
        new_pts[i, 0] = (x - cx) * ca - (y - cy) * sa + cx
        new_pts[i, 1] = (x - cx) * sa + (y - cy) * ca + cy
    return r, new_pts


def random_crop_resize(image, points, scale_range=(0.8, 1.0)):
    w, h = image.size
    scale = random.uniform(*scale_range)
    nw, nh = int(w * scale), int(h * scale)
    new_pts = points.copy() if len(points) > 0 else points
    if scale >= 1.0:
        xs, ys = (nw - w) // 2, (nh - h) // 2
        c = image.resize((nw, nh), Image.BICUBIC).crop((xs, ys, xs + w, ys + h))
        if len(points) > 0:
            new_pts[:, 0] = points[:, 0] * scale - xs
            new_pts[:, 1] = points[:, 1] * scale - ys
    else:
        r = image.resize((nw, nh), Image.BICUBIC)
        c = Image.new("L", (w, h), 0)
        xo, yo = (w - nw) // 2, (h - nh) // 2
        c.paste(r, (xo, yo))
        if len(points) > 0:
            new_pts[:, 0] = points[:, 0] * scale + xo
            new_pts[:, 1] = points[:, 1] * scale + yo
    return c, new_pts


def salt_pepper_noise(image, points, prob=0.005):
    arr = np.array(image)
    m = np.random.random(arr.shape)
    arr[m < prob / 2] = 0
    arr[m > 1 - prob / 2] = 255
    return Image.fromarray(arr), points


def elastic_transform(image, points, alpha=20, sigma=4):
    """- AFM"""
    import cv2

    w, h = image.size
    img_array = np.array(image, dtype=np.float32)
    dx = np.random.RandomState(None).rand(h, w) * 2 - 1
    dy = np.random.RandomState(None).rand(h, w) * 2 - 1
    k = max(3, int(sigma * 3) | 1)
    dx = cv2.GaussianBlur(dx, (k, k), sigma) * alpha
    dy = cv2.GaussianBlur(dy, (k, k), sigma) * alpha
    x_grid, y_grid = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (x_grid + dx).astype(np.float32)
    map_y = (y_grid + dy).astype(np.float32)
    transformed = cv2.remap(
        img_array, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )
    new_points = points.copy() if len(points) > 0 else points
    if len(points) > 0:
        for i, (px, py) in enumerate(points):
            ix, iy = np.clip(int(round(px)), 0, w - 1), np.clip(int(round(py)), 0, h - 1)
            new_points[i, 0] = px + dx[iy, ix]
            new_points[i, 1] = py + dy[iy, ix]
    return Image.fromarray(transformed.astype(np.uint8)), new_points


def cutout(image, points, num_holes=2, max_hole_size=40):
    """(Cutout)- /"""
    w, h = image.size
    img_array = np.array(image)
    for _ in range(num_holes):
        hole_w = random.randint(10, max_hole_size)
        hole_h = random.randint(10, max_hole_size)
        x = random.randint(0, max(1, w - hole_w))
        y = random.randint(0, max(1, h - hole_h))
        img_array[y : y + hole_h, x : x + hole_w] = 0
    return Image.fromarray(img_array), points


def add_gaussian_noise(image, points, std=15):
    """- AFM"""
    img_array = np.array(image)
    noise = np.random.normal(0, std, img_array.shape)
    noisy = np.clip(img_array.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy), points


def gaussian_blur(image, points, radius=1.5):
    """-"""
    return image.filter(ImageFilter.GaussianBlur(radius=radius)), points


def sharpen_image(image, points):
    """-"""
    return image.filter(ImageFilter.SHARPEN), points


def adjust_contrast(image, points, factor):
    """"""
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor), points


def rotate_small(image, points, angle):
    """(90),"""
    w, h = image.size
    rotated = image.rotate(angle, expand=False, fillcolor=0, resample=Image.BICUBIC)
    new_points = points.copy() if len(points) > 0 else points
    if len(points) > 0:
        cx, cy = w / 2, h / 2
        angle_rad = np.radians(angle)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        for i, (x, y) in enumerate(points):
            new_points[i, 0] = (x - cx) * cos_a - (y - cy) * sin_a + cx
            new_points[i, 1] = (x - cx) * sin_a + (y - cy) * cos_a + cy
    return rotated, new_points


def random_crop_resize(image, points, scale_range=(0.8, 1.0)):
    """+"""
    w, h = image.size
    scale = random.uniform(*scale_range)
    new_w, new_h = int(w * scale), int(h * scale)
    new_points = points.copy() if len(points) > 0 else points
    if scale >= 1.0:
        x_start = (new_w - w) // 2
        y_start = (new_h - h) // 2
        resized = image.resize((new_w, new_h), Image.BICUBIC)
        cropped = resized.crop((x_start, y_start, x_start + w, y_start + h))
        if len(points) > 0:
            new_points[:, 0] = points[:, 0] * scale - x_start
            new_points[:, 1] = points[:, 1] * scale - y_start
    else:
        resized = image.resize((new_w, new_h), Image.BICUBIC)
        cropped = Image.new("L", (w, h), 0)
        x_offset = (w - new_w) // 2
        y_offset = (h - new_h) // 2
        cropped.paste(resized, (x_offset, y_offset))
        if len(points) > 0:
            new_points[:, 0] = points[:, 0] * scale + x_offset
            new_points[:, 1] = points[:, 1] * scale + y_offset
    return cropped, new_points


def salt_pepper_noise(image, points, prob=0.005):
    """- AFM /"""
    img_array = np.array(image)
    noise_mask = np.random.random(img_array.shape)
    img_array[noise_mask < prob / 2] = 0
    img_array[noise_mask > 1 - prob / 2] = 255
    return Image.fromarray(img_array), points


def adjust_brightness(image, factor):
    """"""
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)


def augment_single_image(img_path, anno_path, output_img_dir, output_anno_dir, augment_types):
    """"""
    #
    image = Image.open(img_path).convert("L")  #
    w, h = image.size

    #
    if anno_path.exists():
        anno = load_labelme_annotation(anno_path)
        points = anno["points"]
        labels = anno["labels"]
    else:
        points = np.array([])
        labels = []

    results = []
    base_name = img_path.stem

    #
    image.save(output_img_dir / f"{base_name}.png")
    save_labelme_annotation(
        output_anno_dir / f"{base_name}.json", points, labels, f"{base_name}.png", h, w
    )
    results.append(base_name)

    #
    for aug_type in augment_types:
        new_name = f"{base_name}_{aug_type}"
        new_points = points.copy() if len(points) > 0 else points

        if aug_type == "rot90":
            new_img, new_points = rotate_image_and_points(image, points, 90)
        elif aug_type == "rot180":
            new_img, new_points = rotate_image_and_points(image, points, 180)
        elif aug_type == "rot270":
            new_img, new_points = rotate_image_and_points(image, points, 270)
        elif aug_type == "hflip":
            new_img, new_points = flip_image_and_points(image, points, "h")
        elif aug_type == "vflip":
            new_img, new_points = flip_image_and_points(image, points, "v")
        elif aug_type == "rot15":
            new_img, new_points = rotate_small(image, points, random.choice([-15, 15]))
        elif aug_type == "rot30":
            new_img, new_points = rotate_small(image, points, random.choice([-30, 30]))
        elif aug_type == "elastic":
            new_img, new_points = elastic_transform(
                image, points, alpha=random.randint(15, 25), sigma=random.randint(3, 5)
            )
        elif aug_type == "cutout":
            new_img, new_points = cutout(image, points, num_holes=2, max_hole_size=40)
        elif aug_type == "noise":
            new_img, new_points = add_gaussian_noise(image, points, std=random.randint(10, 20))
        elif aug_type == "blur":
            new_img, new_points = gaussian_blur(
                image, points, radius=random.choice([1.0, 1.5, 2.0])
            )
        elif aug_type == "sharpen":
            new_img, new_points = sharpen_image(image, points)
        elif aug_type == "sp_noise":
            new_img, new_points = salt_pepper_noise(image, points, prob=random.uniform(0.003, 0.01))
        elif aug_type == "crop_small":
            new_img, new_points = random_crop_resize(image, points, scale_range=(0.8, 0.95))
        elif aug_type == "bright_more":
            new_img = adjust_brightness(image, 1.5)
            new_points = points.copy() if len(points) > 0 else points
        elif aug_type == "dark_more":
            new_img = adjust_brightness(image, 0.6)
            new_points = points.copy() if len(points) > 0 else points
        elif aug_type == "contrast_high":
            new_img, new_points = adjust_contrast(image, points, 1.5)
        elif aug_type == "contrast_low":
            new_img, new_points = adjust_contrast(image, points, 0.6)
        elif aug_type == "bright":
            new_img = adjust_brightness(image, 1.2)
        elif aug_type == "dark":
            new_img = adjust_brightness(image, 0.8)
        else:
            continue

        #
        new_img.save(output_img_dir / f"{new_name}.png")

        # ,
        if any(k in aug_type for k in ["bright", "dark", "contrast"]):
            save_points = points
        else:
            save_points = new_points

        save_labelme_annotation(
            output_anno_dir / f"{new_name}.json", save_points, labels, f"{new_name}.png", h, w
        )
        results.append(new_name)

    return results


def main(mode="train"):
    """"""
    script_dir = Path(__file__).parent.absolute()
    sub = "val" if mode == "val" else "train"
    sub_aug = sub + "_aug"
    #
    config = {
        # :
        "input_img_dir": str(script_dir / "data" / "processed" / sub),
        "input_anno_dir": str(script_dir / "data" / "annotations" / sub),
        # :
        "output_img_dir": str(script_dir / "data" / "processed" / sub_aug),
        "output_anno_dir": str(script_dir / "data" / "annotations" / sub_aug),
        #
        "augment_types": [
            "rot90",
            "rot180",
            "rot270",
            "hflip",
            "rot15",
            "noise",
            "blur",
            "sharpen",
            "bright",
            "bright_more",
            "dark",
            "dark_more",
            "contrast_high",
            "contrast_low",
        ],
    }

    #
    output_img_dir = Path(config["output_img_dir"])
    output_anno_dir = Path(config["output_anno_dir"])
    output_img_dir.mkdir(parents=True, exist_ok=True)
    output_anno_dir.mkdir(parents=True, exist_ok=True)

    # ()
    for f in output_img_dir.glob("*.png"):
        f.unlink()
    for f in output_anno_dir.glob("*.json"):
        f.unlink()

    input_img_dir = Path(config["input_img_dir"])
    input_anno_dir = Path(config["input_anno_dir"])

    #
    all_img_files = sorted([f for f in input_img_dir.glob("*.png")])
    img_files = []
    for img_path in all_img_files:
        anno_path = input_anno_dir / f"{img_path.stem}.json"
        if anno_path.exists():
            img_files.append(img_path)

    skipped_count = len(all_img_files) - len(img_files)
    print(f"""{len(img_files)} ( {skipped_count} )""")
    print(f""": {config["augment_types"]}""")
    print(f"""{len(config["augment_types"])} + 1() = {len(config["augment_types"]) + 1}""")
    print(f""": {len(img_files) * (len(config["augment_types"]) + 1)}""")
    print("=" * 50)

    all_results = []
    for img_path in tqdm(img_files, desc=""""""):
        anno_path = input_anno_dir / f"{img_path.stem}.json"

        results = augment_single_image(
            img_path, anno_path, output_img_dir, output_anno_dir, config["augment_types"]
        )
        all_results.extend(results)

    print(f"""\n [OK] ! {len(all_results)}""")
    print(f""": {output_img_dir}""")
    print(f""": {output_anno_dir}""")

    #
    img_count = len(list(output_img_dir.glob("*.png")))
    anno_count = len(list(output_anno_dir.glob("*.json")))
    print(f"""\n :""")
    print(f""": {img_count}""")
    print(f""": {anno_count}""")


if __name__ == "__main__":
    import sys

    mode = "train"
    if len(sys.argv) > 1 and sys.argv[1] == "--val":
        mode = "val"
    main(mode)
import random
