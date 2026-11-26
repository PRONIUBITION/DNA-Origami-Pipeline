import cv2
import os
from pathlib import Path


def crop_afm_images(input_dir, output_dir, crop_x=600):
    """Crop AFM images uniformly to a specified width
    crop_x: Cropped width (determined by your image)"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    image_files = (
        list(input_path.glob("*.jpg"))
        + list(input_path.glob("*.png"))
        + list(input_path.glob("*.tif"))
    )

    for img_file in image_files:
        # Read image
        img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"""Unable to read: {img_file}""")
            continue

        # Get current size
        height, width = img.shape

        # Crop uniformly to crop_x width
        # Assuming the valid area is on the left
        if width > crop_x:
            cropped = img[:, :crop_x]  # Take the crop_x pixels on the left
        else:
            cropped = img  # If it is already less than crop_x, leave it as it is.

        # Save
        output_file = output_path / img_file.name
        cv2.imwrite(str(output_file), cropped)

        print(
            f"""Cropped: {img_file.name} {width} x {height} -> {cropped.shape[1]} x {cropped.shape[0]}"""
        )

    print(f"""\n Done! The image has been saved to: {output_dir}""")


# Example of Use
if __name__ == "__main__":
    # Determine the cropping width based on your image
    # You can first check the dimensions of a few images.
    crop_afm_images("""Original Image Directory""", """Directory after cropping""", crop_x=550)
