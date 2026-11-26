import argparse
import base64
import json
import mimetypes
import re
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


PRESETS = {
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-v4-flash-vision-exp",
    },
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen3-vl-plus",
    },
    "glm": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "model": "glm-4v-flash",
    },
}

KEYPOINT_NAMES = [
    "left_dot_1",
    "left_dot_2",
    "left_dot_3",
    "right_dot_1",
    "right_dot_2",
    "right_dot_3",
]

SYSTEM_PROMPT = """You are a strict visual classifier used to evaluate AFM images of DNA origami dimers.
Each cross structure has three points, and different points represent different two-digit random numbers.
The mapping of points and random numbers on each cross-shaped origami structure follows the same rule.
A dimer is formed by two cross-shaped origami structures connected by one of their cross arms.
In other words, there are six points on the dimer, and these points have been color-coded using a traditional visual model.
These six points represent a four-digit random number.
Each side generates two random digits: 00, 01, 10, 11.
Therefore, the four-digit random numbers of the dimer will be 0000, 0001 to 1111, for a total of 16 combinations.

For example: The points on the left cross structure are mapped to 00, and the points on the right cross structure are mapped to 11.
Read from left to right to get"""


def image_paths_from_args(image, input_dir, extensions=(".png", ".jpg", ".jpeg", ".tif", ".tiff")):
    paths = []
    if image:
        paths.append(Path(image))
    if input_dir:
        root = Path(input_dir)
        paths.extend(sorted(p for p in root.iterdir() if p.suffix.lower() in extensions))
    return paths


def encode_image(image_path, overlay_points=None, max_size=1024, output_dir=None):
    image_path = Path(image_path)
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"cannot read image: {image_path}")

    if overlay_points:
        points = load_points(overlay_points)
        color_img = img
        color_map = {
            "left_dot_1": (0, 0, 255),
            "left_dot_2": (0, 255, 0),
            "left_dot_3": (255, 0, 0),
            "right_dot_1": (0, 255, 255),
            "right_dot_2": (255, 0, 255),
            "right_dot_3": (255, 255, 0),
        }
        for point in points:
            label = point.get("label", "")
            x = int(round(float(point["x"])))
            y = int(round(float(point["y"])))
            color = color_map.get(label, (255, 255, 255))
            cv2.circle(color_img, (x, y), max(2, img.shape[0] // 45), color, -1)
            cv2.putText(
                color_img,
                label,
                (x + 4, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                color,
                1,
                cv2.LINE_AA,
            )
        img = color_img

    h, w = img.shape[:2]
    if max_size and max(h, w) > max_size:
        scale = max_size / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        safe_name = image_path.stem + ("_overlay" if overlay_points else "") + ".png"
        cv2.imwrite(str(output_dir / safe_name), img)

    ok, buffer = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError(f"failed to encode image: {image_path}")

    mime = mimetypes.guess_type(image_path.name)[0] or "image/png"
    encoded = base64.b64encode(buffer.tobytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def load_points(points_source):
    source = Path(points_source)
    data = json.loads(source.read_text(encoding="utf-8-sig"))

    if isinstance(data, dict) and "shapes" in data:
        points = []
        for shape in data["shapes"]:
            if shape.get("shape_type") != "point":
                continue
            pt = shape["points"][0]
            points.append(
                {
                    "label": shape.get("label", ""),
                    "x": float(pt[0]),
                    "y": float(pt[1]),
                }
            )
        return points

    if isinstance(data, dict) and "keypoints" in data:
        return data["keypoints"]

    if isinstance(data, list):
        return data

    raise ValueError("points file must contain LabelMe JSON, {'keypoints': [...]}, or a point list")


def collect_references(reference_dir):
    if not reference_dir:
        return []

    root = Path(reference_dir)
    references = []
    for path in sorted(root.iterdir()):
        if path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
            continue
        match = re.match(r"^([01]{4})(?:_.*)?$", path.stem)
        label = match.group(1) if match else path.stem
        references.append((path, label))
    return references


def build_messages(query_image_data, references, overlay_points=None, mapping_image_data=None):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if mapping_image_data:
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": mapping_image_data},
                    },
                    {
                        "type": "text",
                        "text": (
                            "This image defines the mapping between dot positions and the "
                            "two-bit codes. Use it as the classification rule."
                        ),
                    },
                ],
            }
        )

    for ref_path, label in references:
        ref_data = encode_image(ref_path, overlay_points=None)
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": ref_data},
                    },
                    {"type": "text", "text": f"Reference class: {label}"},
                ],
            }
        )

    user_parts = [
        {
            "type": "image_url",
            "image_url": {"url": query_image_data},
        }
    ]
    if overlay_points:
        user_parts.append(
            {
                "type": "text",
                "text": (
                    "The image contains colored overlay dots generated by a keypoint detector. "
                    "Use them to determine the left and right 2-bit codes."
                ),
            }
        )
    else:
        user_parts.append(
            {
                "type": "text",
                "text": "Classify the six bright marker dots and return the four-bit code.",
            }
        )

    messages.append({"role": "user", "content": user_parts})
    return messages


def api_endpoint(base_url):
    base_url = (base_url or "").rstrip("/")
    if base_url.endswith("/chat/completions"):
        return base_url
    return base_url + "/chat/completions"


def call_vlm(messages, api_key, base_url, model, temperature=0.0, timeout=120):
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        api_endpoint(base_url),
        data=data,
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
        raise RuntimeError(f"VLM HTTP {exc.code}: {detail}") from exc

    return json.loads(raw)


def extract_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise ValueError(f"VLM did not return JSON: {text[:200]}")
        return json.loads(match.group(0))


def validate_result(result):
    left = str(result.get("left_code", "")).zfill(2)
    right = str(result.get("right_code", "")).zfill(2)
    if left not in {"00", "01", "10", "11"}:
        left = "??"
    if right not in {"00", "01", "10", "11"}:
        right = "??"
    final = (
        f"{left}{right}"
        if "??" not in (left, right)
        else result.get("final_code", f"{left}{right}")
    )
    result["left_code"] = left
    result["right_code"] = right
    result["final_code"] = final
    return result


def main():
    parser = argparse.ArgumentParser(
        description="OpenAI-compatible VLM classifier for four-bit DNA dimers"
    )
    parser.add_argument("--image", type=str, help="single image path")
    parser.add_argument("--input-dir", type=str, help="directory containing images")
    parser.add_argument(
        "--points", type=str, help="optional keypoint JSON/LabelMe JSON used to draw overlay"
    )
    parser.add_argument(
        "--reference-dir",
        type=str,
        help="optional directory containing reference images named 0000.png ... 1111.png",
    )
    parser.add_argument(
        "--mapping-image", type=str, help="optional image defining the point-to-code mapping rule"
    )
    parser.add_argument("--provider", choices=list(PRESETS.keys()), default="deepseek")
    parser.add_argument("--model", type=str, help="override model name")
    parser.add_argument("--base-url", type=str, help="override base URL")
    parser.add_argument("--api-key", type=str, help="API key; defaults to VLM_API_KEY env var")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--max-size", type=int, default=1024)
    parser.add_argument(
        "--dry-run", action="store_true", help="save payload/visualization without calling API"
    )
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()

    if not args.image and not args.input_dir:
        parser.error("one of --image or --input-dir is required")

    preset = PRESETS[args.provider]
    api_key = args.api_key or __import__("os").getenv("VLM_API_KEY")
    model = args.model or preset["model"]
    base_url = args.base_url or preset["base_url"]

    output_dir = args.output_dir
    if not output_dir:
        output_dir = (
            Path(__file__).parent
            / "outputs"
            / "vlm_classify"
            / datetime.now().strftime("%Y%m%d_%H%M%S")
        )
    output_dir = Path(output_dir)
    viz_dir = output_dir / "visualizations"

    if not api_key and not args.dry_run:
        raise SystemExit("set VLM_API_KEY or pass --api-key")

    image_paths = image_paths_from_args(args.image, args.input_dir)
    if not image_paths:
        raise SystemExit("no images found")

    references = collect_references(args.reference_dir)
    mapping_image_data = (
        encode_image(Path(args.mapping_image), max_size=args.max_size)
        if args.mapping_image
        else None
    )
    results = []
    payloads = []

    for idx, image_path in enumerate(image_paths, start=1):
        print(f"[{idx}/{len(image_paths)}] {image_path}")
        image_data = encode_image(
            image_path,
            overlay_points=args.points,
            max_size=args.max_size,
            output_dir=viz_dir,
        )
        messages = build_messages(
            image_data,
            references,
            overlay_points=args.points,
            mapping_image_data=mapping_image_data,
        )

        if args.dry_run:
            payloads.append({"image": str(image_path), "messages": messages})
            print("  dry-run: payload ready")
            continue

        raw_response = call_vlm(
            messages,
            api_key=api_key,
            base_url=base_url,
            model=model,
            timeout=args.timeout,
        )
        content = raw_response["choices"][0]["message"]["content"]
        parsed = validate_result(extract_json(content))
        parsed["source_image"] = str(image_path)
        parsed["model"] = model
        parsed["provider"] = args.provider
        results.append(parsed)

        classified_img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if classified_img is not None:
            viz_dir.mkdir(parents=True, exist_ok=True)
            h, w = classified_img.shape[:2]
            text = str(parsed.get("final_code", "??"))
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.35
            thickness = 1
            (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
            text_x = (w - text_width) // 2
            text_y = h - 10
            cv2.rectangle(
                classified_img,
                (text_x - 2, text_y - text_height - 2),
                (text_x + text_width + 2, text_y + 2),
                (0, 0, 0),
                -1,
            )
            cv2.putText(
                classified_img,
                text,
                (text_x, text_y),
                font,
                font_scale,
                (0, 255, 255),
                thickness,
            )
            cv2.imwrite(
                str(viz_dir / f"{image_path.stem}_vlm.png"),
                classified_img,
            )

        print(f"  {parsed.get('final_code', '??')}")

        if idx < len(image_paths):
            time.sleep(0.2)

    output_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        with open(output_dir / "dry_run_payloads.json", "w", encoding="utf-8") as f:
            json.dump(payloads, f, indent=2, ensure_ascii=False)
        print(f"Dry-run payloads: {output_dir / 'dry_run_payloads.json'}")
    else:
        with open(output_dir / "classifications.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results: {output_dir / 'classifications.json'}")


if __name__ == "__main__":
    main()
