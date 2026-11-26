import json
import random
import shutil
from pathlib import Path


ROOT = Path(__file__).parent
SOURCE_DIR = ROOT / "data" / "fourbits" / "segments&annotations"
DATA_DIR = ROOT / "data" / "fourbits"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test_set"
TRAIN_ANNO_DIR = DATA_DIR / "annotations" / "train"
TEST_ANNO_DIR = DATA_DIR / "annotations" / "test_set"

COPY_COUNT = 5
TEST_RATIO = 0.2
SEED = 42


def clean_dir(path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def collect_annotated_items():
    items = []
    for json_path in sorted(SOURCE_DIR.glob("*.json")):
        if json_path.name.startswith("_"):
            continue
        image_path = json_path.with_suffix(".png")
        if not image_path.exists():
            continue
        data = json.loads(json_path.read_text(encoding="utf-8-sig"))
        shapes = data.get("shapes", [])
        if not shapes:
            continue
        items.append((image_path, json_path, len(shapes)))
    return items


def copy_instance(image_path, json_path, image_dir, anno_dir, base_name, copy_index):
    suffix = f"_copy{copy_index}"
    new_stem = f"{image_path.stem}{suffix}"
    new_image = image_dir / f"{new_stem}.png"
    new_json = anno_dir / f"{new_stem}.json"

    shutil.copy2(image_path, new_image)

    data = json.loads(json_path.read_text(encoding="utf-8-sig"))
    data["imagePath"] = new_image.name
    with open(new_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    items = collect_annotated_items()
    if not items:
        raise SystemExit("no annotated fourbits instances found")

    random.seed(SEED)
    items.sort(key=lambda item: item[0].stem)
    test_count = max(1, int(round(len(items) * TEST_RATIO)))
    test_items = random.sample(items, test_count)
    test_stems = {item[0].stem for item in test_items}
    train_items = items  # test set is also kept in training for intentional overfitting

    for path in (TRAIN_DIR, TEST_DIR, TRAIN_ANNO_DIR, TEST_ANNO_DIR):
        clean_dir(path)

    manifest = {
        "source_dir": str(SOURCE_DIR),
        "copy_count": COPY_COUNT,
        "test_ratio": TEST_RATIO,
        "seed": SEED,
        "annotated_instances": len(items),
        "train_unique": len(train_items),
        "test_unique": len(test_items),
        "train_copies": len(train_items) * COPY_COUNT,
        "test_copies": len(test_items) * COPY_COUNT,
        "test_stems": sorted(test_stems),
        "items": [],
    }

    for image_path, json_path, shape_count in train_items:
        for copy_index in range(COPY_COUNT):
            copy_instance(
                image_path,
                json_path,
                TRAIN_DIR,
                TRAIN_ANNO_DIR,
                image_path.stem,
                copy_index,
            )
        manifest["items"].append(
            {
                "stem": image_path.stem,
                "shape_count": shape_count,
                "in_train": True,
                "in_test": image_path.stem in test_stems,
            }
        )

    for image_path, json_path, shape_count in test_items:
        for copy_index in range(COPY_COUNT):
            copy_instance(
                image_path,
                json_path,
                TEST_DIR,
                TEST_ANNO_DIR,
                image_path.stem,
                copy_index,
            )

    with open(DATA_DIR / "overfit_split_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Annotated instances: {len(items)}")
    print(f"Train unique/copies: {len(train_items)}/{len(train_items) * COPY_COUNT}")
    print(f"Test unique/copies: {len(test_items)}/{len(test_items) * COPY_COUNT}")
    print(f"Train dir: {TRAIN_DIR}")
    print(f"Test dir: {TEST_DIR}")


if __name__ == "__main__":
    main()
