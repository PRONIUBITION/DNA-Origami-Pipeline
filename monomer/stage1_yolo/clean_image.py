import cv2
import numpy as np
from pathlib import Path
from settings import data_path, output_path

# ========== Configuration Path ==========
INPUT_IMG_DIR = data_path("stage1", "images")
INPUT_MASK_DIR = data_path("stage1", "masks")
OUTPUT_CLEAN_DIR = output_path("stage1", "clean_images")

SAVE_VISUALIZATION = True
OUTPUT_VIS_DIR = output_path("stage1", "visual_check")

# ========== Added: Dilation Parameters ==========
# This value determines how many pixels the mask expands outward, ensuring that it covers the small white dots but does not excessively cover the background.
# If the small white dot is far from the cross, increase the value (e.g., 40 or 50); if it is close, decrease the value (e.g., 20).
DILATE_PIXELS = 6


# ==============================


def apply_mask_with_dilate():
    """Batch processing: After the mask is dilated, the background is blackened, and the small white dots are retained."""

    clean_dir = Path(OUTPUT_CLEAN_DIR)
    clean_dir.mkdir(parents=True, exist_ok=True)

    if SAVE_VISUALIZATION:
        vis_dir = Path(OUTPUT_VIS_DIR)
        vis_dir.mkdir(parents=True, exist_ok=True)

    img_files = list(Path(INPUT_IMG_DIR).glob("*_img.png"))
    total = len(img_files)

    print(f"""{total} images found. Starting processing...""")
    print(
        f"""Mask dilation radius: {DILATE_PIXELS} pixels (retain the small white dot at the end of the arm)"""
    )

    for i, img_path in enumerate(img_files, 1):
        stem = img_path.stem.replace("_img", "")
        mask_path = Path(INPUT_MASK_DIR) / f"{stem}_mask.npy"

        if not mask_path.exists():
            continue

        try:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            mask = np.load(str(mask_path))

            if img.shape != mask.shape:
                mask = cv2.resize(
                    mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST
                )

            # ========== Key modification: Dilate Mask! ==========
            # Create a dilation kernel (circular, size DILATE_PIXELS)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (DILATE_PIXELS, DILATE_PIXELS))

            # Dilate mask: Expand the cross shape outward to cover the small white dots at the ends of the arms.
            mask_dilated = cv2.dilate(mask, kernel, iterations=1)

            # Convert back to 0-1 format (may be 0-255 after dilation)
            mask_dilated = (mask_dilated > 0).astype(np.float32)
            # ==========================================

            # Use the dilated mask to darken the background (retain the cross + small white dot area)
            clean_img = (img * mask_dilated).astype(np.uint8)

            output_path = clean_dir / f"{stem}_clean.png"
            cv2.imwrite(str(output_path), clean_img)

            # Save the dilated mask for later use
            np.save(str(Path(INPUT_MASK_DIR) / f"{stem}_mask_dilated.npy"), mask_dilated)

            if SAVE_VISUALIZATION and i <= 40:
                # Generate a comparison of three images: Original image | Original mask effect | Effect after dilation
                original_masked = (img * mask).astype(
                    np.uint8
                )  # Original mask (may cut off small white dots)

                comparison = np.hstack([img, original_masked, clean_img])

                # Small font size label
                cv2.putText(
                    comparison, "Original", (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,), 1
                )
                cv2.putText(
                    comparison,
                    "Mask",
                    (img.shape[1] + 10, 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (255,),
                    1,
                )
                cv2.putText(
                    comparison,
                    "Dilated",
                    (img.shape[1] * 2 + 10, 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (255,),
                    1,
                )

                vis_path = vis_dir / f"{stem}_compare.png"
                cv2.imwrite(str(vis_path), comparison)

                # Print a warning if white spots that may have been cut off are detected.
                if (
                    np.sum(original_masked) < np.sum(clean_img) * 0.9
                ):  # The number of bright pixels increases significantly after dilation.
                    print(
                        f"""[!] {stem}: The original mask may have removed some bright spots, which have been repaired by dilation."""
                    )

            print(f"""[ {i} / {total} ] {stem} : Processed (dilation {DILATE_PIXELS} px)""")

        except Exception as e:
            print(f"""[X] {stem}: Failed - {e}""")

    print(f"""\n [] Done! The dilated clean image is at: {OUTPUT_CLEAN_DIR}""")
    print(
        f"""Check the comparison image (see if the white dots are retained in column 3): {OUTPUT_VIS_DIR}"""
    )


if __name__ == "__main__":
    apply_mask_with_dilate()
