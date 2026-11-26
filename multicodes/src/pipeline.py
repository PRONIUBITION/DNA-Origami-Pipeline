import argparse
import base64
import html
import json
import re
import urllib.error
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


QWEN_SYSTEM_PROMPT = """You are a strict visual classifier for four-bit
binary codes on cross-shaped DNA origami structures.

Each cross has horizontal and vertical arms. Up to nine designed sites can
bind streptavidin and appear as small white dots. An orange marker near the
upper-left side of the vertical arm defines the orientation.

Read the bits in this order:
1. Vertical arm, left side: top = 1, bottom = 0.
2. Vertical arm, right side: top = 1, bottom = 0.
3. Horizontal arm, upper side: right = 1, left = 0.
4. Horizontal arm, lower side: right = 1, left = 0.

Use the orange marker to determine orientation. Do not infer orientation from
an arbitrary image rotation.

Return one valid JSON object only, with no Markdown or explanation:
{"code":"1010","confidence":0.95,"reason":"brief explanation"}

The code must be a four-character binary string from 0000 to 1111."""


def imread_unicode(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path, image):
    ext = Path(path).suffix.lower() or ".png"
    params = [cv2.IMWRITE_PNG_COMPRESSION, 3] if ext == ".png" else []
    ok, buffer = cv2.imencode(ext, image, params)
    if not ok:
        raise RuntimeError(f"Failed to encode image: {path}")
    with open(path, "wb") as f:
        f.write(buffer)


def image_to_data_url(image):
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("Failed to encode image for Qwen")
    encoded = base64.b64encode(buffer.tobytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def resize_for_api(image, max_size):
    h, w = image.shape[:2]
    if max_size and max(h, w) > max_size:
        scale = max_size / max(h, w)
        image = cv2.resize(
            image,
            (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return image


def normalize_binary(gray):
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if binary.mean() > 127:
        binary = cv2.bitwise_not(binary)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    return binary


def expand_box(box, image_shape, padding):
    x1, y1, x2, y2 = box
    h, w = image_shape[:2]
    return (
        max(0, x1 - padding),
        max(0, y1 - padding),
        min(w, x2 + padding),
        min(h, y2 + padding),
    )


def segment_by_erosion(image, erode_radius, min_area, padding):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = normalize_binary(gray)
    ksize = max(3, int(erode_radius) * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    eroded = cv2.erode(binary, kernel, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(eroded, connectivity=8)
    boxes = []
    debug_mask = np.zeros_like(binary)

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        boxes.append(
            expand_box(
                (x, y, x + w, y + h),
                image.shape,
                padding + erode_radius,
            )
        )
        debug_mask[labels == label] = 255

    boxes.sort(key=lambda b: (b[1], b[0]))
    return boxes, binary, debug_mask


def segment_crosses(image, mode, erode_radius, min_area, padding):
    if mode == "single":
        h, w = image.shape[:2]
        return [(0, 0, w, h)], None, None

    boxes, binary, debug_mask = segment_by_erosion(image, erode_radius, min_area, padding)
    if mode == "auto" and len(boxes) <= 1:
        h, w = image.shape[:2]
        boxes = [(0, 0, w, h)]
    return boxes, binary, debug_mask


def extract_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
            if isinstance(value, dict) and "code" in value:
                return value
        except json.JSONDecodeError:
            continue

    code_match = re.search(r'"code"\s*:\s*"?([01]{4})"?', text)
    if code_match:
        confidence_match = re.search(r'"confidence"\s*:\s*([0-9.]+)', text)
        return {
            "code": code_match.group(1),
            "confidence": (float(confidence_match.group(1)) if confidence_match else None),
            "reason": "Parsed code from non-standard Qwen response.",
        }

    raise ValueError(f"Qwen did not return JSON: {text[:200]}")


def extract_pptx_reference(reference_dir, extract_dir):
    reference_dir = Path(reference_dir)
    extract_dir = Path(extract_dir)
    media_dir = extract_dir / "media"
    media_dir.mkdir(parents=True, exist_ok=True)

    texts = []
    image_paths = []
    for text_path in sorted(reference_dir.glob("*")):
        if text_path.suffix.lower() in {".md", ".txt"}:
            texts.append(f"[{text_path.name}]\n{text_path.read_text(encoding='utf-8')}")
    for image_path in sorted(reference_dir.glob("*")):
        if image_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
            image_paths.append(image_path)

    for pptx_path in sorted(reference_dir.glob("*.pptx")):
        with zipfile.ZipFile(pptx_path) as archive:
            slide_names = sorted(
                (
                    name
                    for name in archive.namelist()
                    if re.fullmatch(r"ppt/slides/slide\d+\.xml", name)
                ),
                key=lambda name: int(re.search(r"(\d+)", Path(name).stem).group(1)),
            )
            for slide_name in slide_names:
                xml = archive.read(slide_name).decode("utf-8", errors="ignore")
                slide_text = " ".join(
                    html.unescape(value)
                    for value in re.findall(r"<a:t>(.*?)</a:t>", xml, flags=re.S)
                )
                if slide_text.strip():
                    texts.append(f"[{pptx_path.name} - {Path(slide_name).stem}]\n{slide_text}")

            for media_name in sorted(
                (
                    name
                    for name in archive.namelist()
                    if name.startswith("ppt/media/")
                    and Path(name).suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff")
                ),
                key=lambda name: int(re.search(r"(\d+)", Path(name).stem).group(1)),
            ):
                output_path = media_dir / Path(media_name).name
                output_path.write_bytes(archive.read(media_name))
                image_paths.append(output_path)

    document = "\n\n".join(texts)
    (extract_dir / "document.txt").write_text(document, encoding="utf-8")

    if image_paths:
        sheet = build_reference_sheet(image_paths, extract_dir / "reference_sheet.png")
        return document, [sheet]
    return document, []


def build_reference_sheet(image_paths, output_path, cell_size=512, columns=3):
    loaded = []
    for image_path in image_paths:
        image = imread_unicode(image_path)
        if image is not None:
            loaded.append(image)
    if not loaded:
        return None

    rows = (len(loaded) + columns - 1) // columns
    sheet = np.full(
        (rows * cell_size, columns * cell_size, 3),
        255,
        dtype=np.uint8,
    )

    for index, image in enumerate(loaded):
        h, w = image.shape[:2]
        scale = min((cell_size - 20) / w, (cell_size - 20) / h)
        resized = cv2.resize(
            image,
            (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
        row = index // columns
        col = index % columns
        y0 = row * cell_size + (cell_size - resized.shape[0]) // 2
        x0 = col * cell_size + (cell_size - resized.shape[1]) // 2
        sheet[
            y0 : y0 + resized.shape[0],
            x0 : x0 + resized.shape[1],
        ] = resized
        cv2.putText(
            sheet,
            f"ref {index + 1}",
            (col * cell_size + 10, row * cell_size + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

    imwrite_unicode(output_path, sheet)
    return output_path


def call_qwen(
    image,
    api_key,
    base_url,
    model,
    timeout,
    reference_text="",
    reference_images=None,
    enable_thinking=False,
):
    reference_images = reference_images or []
    system_prompt = QWEN_SYSTEM_PROMPT
    if reference_text.strip():
        system_prompt += """\n \n : \n""" + reference_text

    user_content = []
    for index, reference_image_path in enumerate(reference_images, start=1):
        reference_image = imread_unicode(reference_image_path)
        if reference_image is None:
            continue
        reference_image = resize_for_api(reference_image, 1024)
        user_content.append(
            {
                "type": "image_url",
                "image_url": {"url": image_to_data_url(reference_image)},
            }
        )
        user_content.append(
            {
                "type": "text",
                "text": f"""{index} ,""",
            }
        )

    user_content.extend(
        [
            {
                "type": "image_url",
                "image_url": {"url": image_to_data_url(image)},
            },
            {
                "type": "text",
                "text": """""",
            },
        ]
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.0,
        "enable_thinking": enable_thinking,
    }

    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Qwen HTTP {exc.code}: {detail}") from exc

    data = json.loads(raw)
    content = data["choices"][0]["message"]["content"]
    result = extract_json(content)
    code = str(result.get("code", ""))
    if not re.fullmatch(r"[01]{4}", code):
        raise ValueError(f"Invalid code from Qwen: {code}")
    return result


def draw_result(image, box, result, index):
    x1, y1, x2, y2 = box
    vis = image.copy()
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 1)

    code = result.get("code", "????")
    text = f"{index}: {code}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    thickness = 1
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    tx = x1
    ty = max(th + 4, y1 - 4)
    cv2.rectangle(vis, (tx, ty - th - 3), (tx + tw + 4, ty + 3), (0, 0, 0), -1)
    cv2.putText(
        vis,
        text,
        (tx + 2, ty),
        font,
        font_scale,
        (0, 255, 255),
        thickness,
    )
    return vis


def main():
    parser = argparse.ArgumentParser(
        description="Split DNA origami cross chains and classify each cross with Qwen."
    )
    parser.add_argument("--input", required=True, help="input image or directory")
    parser.add_argument(
        "--output",
        default=None,
        help="output directory; defaults to outputs/<timestamp>",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "single", "chain"],
        default="auto",
        help="single: whole image is one cross; chain: split; auto: split if multiple regions",
    )
    parser.add_argument("--erode-radius", type=int, default=4)
    parser.add_argument("--min-area", type=int, default=800)
    parser.add_argument("--padding", type=int, default=12)
    parser.add_argument("--max-size", type=int, default=1536)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--config", default=str(root / "config" / "config.json"))
    parser.add_argument(
        "--reference-dir",
        default=str(root / "assets" / "reference"),
        help="directory containing PPTX or reference images",
    )
    parser.add_argument("--no-reference", action="store_true")
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="enable Qwen reasoning mode (slower, sometimes more accurate)",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    config = {}
    config_path = Path(args.config)
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8-sig"))

    api_key = args.api_key or config.get("api_key")
    if not api_key:
        api_key = __import__("os").environ.get("DASHSCOPE_API_KEY") or __import__("os").environ.get(
            "QWEN_API_KEY"
        )
    model = args.model or config.get("model") or "qwen3.8-max"
    base_url = (
        args.base_url
        or config.get("base_url")
        or "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    if not api_key and not args.dry_run:
        raise SystemExit("Set api_key in config.json, set DASHSCOPE_API_KEY, or pass --api-key")

    input_path = Path(args.input)
    if input_path.is_file():
        image_paths = [input_path]
    elif input_path.is_dir():
        image_paths = sorted(
            p
            for p in input_path.iterdir()
            if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff")
        )
    else:
        raise SystemExit(f"Input path does not exist: {input_path}")

    if args.limit:
        image_paths = image_paths[: args.limit]

    output_dir = args.output
    if output_dir is None:
        output_dir = (
            Path(__file__).resolve().parents[1]
            / "outputs"
            / datetime.now().strftime("%Y%m%d_%H%M%S")
        )
    output_dir = Path(output_dir)
    crops_dir = output_dir / "crops"
    masks_dir = output_dir / "masks"
    viz_dir = output_dir / "visualizations"
    for directory in (crops_dir, masks_dir, viz_dir):
        directory.mkdir(parents=True, exist_ok=True)

    reference_text = ""
    reference_images = []
    if not args.no_reference:
        reference_text, reference_images = extract_pptx_reference(
            args.reference_dir, output_dir / "reference"
        )
        print(f"Reference: {len(reference_text)} chars, {len(reference_images)} images")

    all_results = []
    for image_path in image_paths:
        image = imread_unicode(image_path)
        if image is None:
            print(f"[skip] cannot read: {image_path}")
            continue

        h, w = image.shape[:2]
        if args.max_size and max(h, w) > args.max_size:
            scale = args.max_size / max(h, w)
            image = cv2.resize(
                image,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_AREA,
            )

        boxes, binary, debug_mask = segment_crosses(
            image,
            args.mode,
            args.erode_radius,
            args.min_area,
            args.padding,
        )

        if binary is not None:
            imwrite_unicode(masks_dir / f"{image_path.stem}_binary.png", binary)
        if debug_mask is not None:
            imwrite_unicode(masks_dir / f"{image_path.stem}_eroded.png", debug_mask)

        image_results = []
        vis = image.copy()

        for index, box in enumerate(boxes, start=1):
            x1, y1, x2, y2 = box
            crop = image[y1:y2, x1:x2]
            crop_name = f"{image_path.stem}_cross_{index:02d}.png"
            imwrite_unicode(crops_dir / crop_name, crop)

            if args.dry_run:
                result = {"code": "DRY", "confidence": None, "reason": "dry-run"}
            else:
                try:
                    result = call_qwen(
                        crop,
                        api_key,
                        base_url,
                        model,
                        args.timeout,
                        reference_text=reference_text,
                        reference_images=reference_images,
                        enable_thinking=args.enable_thinking,
                    )
                except Exception as exc:
                    result = {
                        "code": "ERROR",
                        "confidence": None,
                        "reason": str(exc),
                    }

            item = {
                "image": str(image_path),
                "cross_index": index,
                "bbox": [int(v) for v in box],
                "crop": crop_name,
                **result,
            }
            image_results.append(item)
            all_results.append(item)
            vis = draw_result(vis, box, result, index)
            print(f"{image_path.name} cross {index}: {result.get('code', 'ERROR')}")

        imwrite_unicode(viz_dir / f"{image_path.stem}_result.png", vis)
        with open(
            output_dir / f"{image_path.stem}_results.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(image_results, f, indent=2, ensure_ascii=False)

    with open(output_dir / "all_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"Processed {len(image_paths)} images")
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
