"""Intelligent Uniform Cropping Tool for DNA Origami AFM Images
Function: Automatically analyzes the valid area of the image, standardizes the cropping size, and keeps the color unchanged.
Author: AI Assistant
Version: 1.0"""

import cv2
import numpy as np
import os
import sys
from pathlib import Path
import argparse
from tqdm import tqdm
import json
from settings import data_path

input_path = data_path("cropping", "input")
output_path = data_path("cropping", "output")


class DNACropper:
    """DNA Origami Image Smart Cropper"""

    def __init__(self, verbose=True):
        self.verbose = verbose
        self.crop_params = None  # Save cropping parameters for annotation adjustments

    def log(self, message):
        """Print Log"""
        if self.verbose:
            print(message)

    def read_image_preserve_color(self, image_path):
        """Read the image and retain the original color mode
        Returns: Image array and color mode information"""
        # Attempt to read as color
        img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

        if img is None:
            raise ValueError(f"""Unable to read image: {image_path}""")

        # Get image information
        if len(img.shape) == 2:
            color_mode = "GRAY"
        elif len(img.shape) == 3:
            if img.shape[2] == 3:
                color_mode = "BGR"
            elif img.shape[2] == 4:
                color_mode = "BGRA"
            else:
                color_mode = "UNKNOWN"
        else:
            color_mode = "UNKNOWN"

        self.log(
            f"""Read: {os.path.basename(image_path)} - Size: {img.shape} - Mode: {color_mode}"""
        )
        return img, color_mode

    def analyze_sample_images(self, image_paths, sample_size=10, threshold_percentile=5):
        """Analyze sample images to determine valid areas"""
        self.log(f"""Analyzing {min(len(image_paths), sample_size)} sample images...""")

        valid_regions = []
        color_modes = set()

        # Randomly select samples (or take the first few in order)
        sample_indices = np.random.choice(
            len(image_paths), min(sample_size, len(image_paths)), replace=False
        )

        for idx in tqdm(sample_indices, desc="""Analyze samples"""):
            img_path = image_paths[idx]

            try:
                img, color_mode = self.read_image_preserve_color(img_path)
                color_modes.add(color_mode)

                # Convert to grayscale for analysis (without changing the original image)
                if len(img.shape) == 3:
                    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                else:
                    img_gray = img.copy()

                height, width = img_gray.shape

                # Calculate adaptive threshold
                if np.std(img_gray) < 5:  # Low-contrast image
                    threshold = np.percentile(img_gray, threshold_percentile) + 5
                else:
                    threshold = np.percentile(img_gray, threshold_percentile) + 10

                # Detecting the effective area
                # 1. Horizontal projection
                col_sum = np.sum(img_gray > threshold, axis=0)
                if len(col_sum) > 0:
                    valid_cols = np.where(col_sum > height * 0.02)[
                        0
                    ]  # At least 2% of the height contains content
                else:
                    valid_cols = np.array([])

                # 2. Vertical projection
                row_sum = np.sum(img_gray > threshold, axis=1)
                if len(row_sum) > 0:
                    valid_rows = np.where(row_sum > width * 0.02)[
                        0
                    ]  # At least 2% of the width contains content
                else:
                    valid_rows = np.array([])

                if len(valid_cols) > 0 and len(valid_rows) > 0:
                    left = valid_cols[0]
                    right = valid_cols[-1]
                    top = valid_rows[0]
                    bottom = valid_rows[-1]

                    # Add margin (5%)
                    margin_h = int((right - left) * 0.05)
                    margin_v = int((bottom - top) * 0.05)

                    left = max(0, left - margin_h)
                    right = min(width, right + margin_h)
                    top = max(0, top - margin_v)
                    bottom = min(height, bottom + margin_v)

                    valid_regions.append(
                        {
                            "left": left,
                            "right": right,
                            "top": top,
                            "bottom": bottom,
                            "width": width,
                            "height": height,
                        }
                    )

            except Exception as e:
                self.log(f"""Error analyzing {os.path.basename(img_path)}: {e}""")
                continue

        if not valid_regions:
            raise ValueError("""No valid areas detected. Please check the image quality.""")

        self.log(f"""Color Mode Distribution: {color_modes}""")
        return valid_regions

    def calculate_unified_crop_params(self, valid_regions, min_margin_ratio=0.02):
        """Calculate unified cropping parameters"""
        # Extract all boundaries
        lefts = [r["left"] for r in valid_regions]
        rights = [r["right"] for r in valid_regions]
        tops = [r["top"] for r in valid_regions]
        bottoms = [r["bottom"] for r in valid_regions]
        widths = [r["width"] for r in valid_regions]
        heights = [r["height"] for r in valid_regions]

        # Calculate the most conservative crop (ensuring all images are included)
        # Use the 5% quantile as a safety margin
        safe_left = int(np.percentile(lefts, 5))
        safe_right = int(np.percentile(rights, 95))
        safe_top = int(np.percentile(tops, 5))
        safe_bottom = int(np.percentile(bottoms, 95))

        # Convert to ratio (relative to image size)
        median_width = int(np.median(widths))
        median_height = int(np.median(heights))

        # Ensure that the cropping area is valid
        if safe_left >= safe_right or safe_top >= safe_bottom:
            self.log(
                """Warning: The automatically detected area is invalid. Using default cropping."""
            )
            # Default crop: Retain the central 90% area
            safe_left = int(median_width * 0.05)
            safe_right = int(median_width * 0.95)
            safe_top = int(median_height * 0.05)
            safe_bottom = int(median_height * 0.95)

        # Convert to cropping parameters (number of pixels to crop from the edge)
        crop_left = safe_left
        crop_right = median_width - safe_right

        crop_top = safe_top
        crop_bottom = median_height - safe_bottom

        # Ensure that at least a certain margin is retained
        min_margin_x = int(median_width * min_margin_ratio)
        min_margin_y = int(median_height * min_margin_ratio)

        crop_left = max(crop_left, min_margin_x)
        crop_right = max(crop_right, min_margin_x)
        extra_right_crop = 160  # Crop an additional 200 pixels (adjust based on your image)
        crop_right = crop_right + extra_right_crop
        crop_top = max(crop_top, min_margin_y)
        crop_bottom = max(crop_bottom, min_margin_y)
        extra_bottom_crop = 40  # Crop an additional 40 pixels (adjust according to your image)
        crop_bottom = crop_bottom + extra_bottom_crop

        self.crop_params = {
            "crop_left": crop_left,
            "crop_right": crop_right,
            "crop_top": crop_top,
            "crop_bottom": crop_bottom,
            "original_median_size": (median_width, median_height),
            "cropped_size": (
                median_width - crop_left - crop_right,
                median_height - crop_top - crop_bottom,
            ),
        }

        self.log(f"""\n === Cropping parameter calculation complete ===""")
        self.log(f"""Median size of original image: {median_width} x {median_height}""")
        self.log(f"""Left crop: {crop_left} pixels""")
        self.log(f"""Right crop: {crop_right} pixels""")
        self.log(f"""Crop from top: {crop_top} pixels""")
        self.log(f"""Bottom crop: {crop_bottom} pixels""")
        self.log(
            f"""Size after cropping: {self.crop_params["cropped_size"][0]} x {self.crop_params["cropped_size"][1]}"""
        )

        return self.crop_params

    def apply_crop_to_image(self, img, crop_params):
        """Apply cropping to a single image"""
        height, width = img.shape[:2]

        # If the image size differs too much from the median size used to calculate the parameters, adjust proportionally
        median_width, median_height = crop_params["original_median_size"]

        if (
            abs(width - median_width) > median_width * 0.2
            or abs(height - median_height) > median_height * 0.2
        ):
            # If the size difference exceeds 20%, adjust the cropping parameters proportionally.
            scale_x = width / median_width
            scale_y = height / median_height

            crop_left = int(crop_params["crop_left"] * scale_x)
            crop_right = int(crop_params["crop_right"] * scale_x)
            crop_top = int(crop_params["crop_top"] * scale_y)
            crop_bottom = int(crop_params["crop_bottom"] * scale_y)
        else:
            crop_left = crop_params["crop_left"]
            crop_right = crop_params["crop_right"]
            crop_top = crop_params["crop_top"]
            crop_bottom = crop_params["crop_bottom"]

        # Calculate the cropping area
        left = crop_left
        right = width - crop_right
        top = crop_top
        bottom = height - crop_bottom

        # Ensure that the cropping area is valid
        if left >= right or top >= bottom:
            self.log(f"""Warning: Cropping parameters are invalid. Using the central 90% area.""")
            left = int(width * 0.05)
            right = int(width * 0.95)
            top = int(height * 0.05)
            bottom = int(height * 0.95)

        # Perform cropping
        if len(img.shape) == 3:  # Color image
            cropped = img[top:bottom, left:right, :]
        else:  # Grayscale image
            cropped = img[top:bottom, left:right]

        return cropped

    def batch_crop_images(self, input_dir, output_dir, sample_size=10, force_recalculate=False):
        """Batch Crop Images"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)

        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)

        # Get all image files
        image_extensions = [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"]
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(input_path.glob(f"*{ext}")))
            image_files.extend(list(input_path.glob(f"*{ext.upper()}")))

        if not image_files:
            raise ValueError(f"""No image files found in {input_dir}""")

        self.log(f"""{len(image_files)} images found""")

        # Check if cropping parameters already exist
        params_file = output_path / "crop_params.json"
        if params_file.exists() and not force_recalculate:
            with open(params_file, "r") as f:
                self.crop_params = json.load(f)
            self.log("""Use saved cropping parameters""")
        else:
            # Analyze the image and calculate the cropping parameters
            valid_regions = self.analyze_sample_images(image_files, sample_size)
            self.calculate_unified_crop_params(valid_regions)

            # Save Parameters
            with open(params_file, "w") as f:
                json.dump(self.crop_params, f, indent=2)

        # Batch Crop
        self.log(f"""\n Start batch cropping...""")
        success_count = 0
        failed_images = []

        for img_file in tqdm(image_files, desc="""Crop image"""):
            try:
                # Read Image (Preserve Color)
                img, color_mode = self.read_image_preserve_color(img_file)

                # Apply Crop
                cropped = self.apply_crop_to_image(img, self.crop_params)

                # Save image (preserve original format and quality)
                output_file = output_path / img_file.name

                # Determine save parameters based on the original file extension
                ext = img_file.suffix.lower()
                if ext in [".jpg", ".jpeg"]:
                    cv2.imwrite(str(output_file), cropped, [cv2.IMWRITE_JPEG_QUALITY, 95])
                elif ext in [".png"]:
                    cv2.imwrite(str(output_file), cropped, [cv2.IMWRITE_PNG_COMPRESSION, 3])
                elif ext in [".tif", ".tiff"]:
                    cv2.imwrite(str(output_file), cropped)
                else:
                    cv2.imwrite(str(output_file), cropped)

                success_count += 1

            except Exception as e:
                error_msg = f"""Error processing {img_file.name}: {str(e)}"""
                self.log(error_msg)
                failed_images.append((img_file.name, str(e)))

        # Generate Report
        self.log(f"""\n === Cropping completed ===""")
        self.log(f"""Successfully processed: {success_count} / {len(image_files)}""")
        self.log(f"""Output Directory: {output_dir}""")

        if failed_images:
            self.log(f"""Failed images: {len(failed_images)}""")
            for img_name, error in failed_images[:5]:  # Show only the first 5 errors
                self.log(f"  {img_name}: {error}")

            # Save Error Log
            error_log = output_path / "crop_errors.log"
            with open(error_log, "w") as f:
                f.write("""List of failed images: \n""")
                for img_name, error in failed_images:
                    f.write(f"{img_name}: {error}\n")

        # Save Cropping Information
        info_file = output_path / "crop_info.txt"
        with open(info_file, "w") as f:
            f.write("""DNA Origami Image Cropping Report \n""")
            f.write("=" * 50 + "\n")
            f.write(f"""Input directory: {input_dir} \n""")
            f.write(f"""Output Directory: {output_dir} \n""")
            f.write(f"""Total number of images: {len(image_files)} \n""")
            f.write(f"""Successfully cropped: {success_count} \n""")
            f.write(f"""Number of failures: {len(failed_images)} \n \n""")

            f.write("""Cropping parameters: \n""")
            for key, value in self.crop_params.items():
                f.write(f"  {key}: {value}\n")

        return self.crop_params, failed_images

    def preview_crop(self, image_path, crop_params=None):
        """Preview cropping effect"""
        if crop_params is None:
            crop_params = self.crop_params

        if crop_params is None:
            raise ValueError(
                """Please calculate the cropping parameters or provide the parameters first."""
            )

        img, color_mode = self.read_image_preserve_color(image_path)
        cropped = self.apply_crop_to_image(img, crop_params)

        # Create a comparison image
        if len(img.shape) == 3:
            img_display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cropped_display = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        else:
            img_display = img
            cropped_display = cropped

        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original Image
        axes[0].imshow(img_display, cmap="gray" if color_mode == "GRAY" else None)
        axes[0].set_title(f"""Original image \n {img.shape[1]} x {img.shape[0]}""")
        axes[0].axis("off")

        # Mark the cropping area
        axes[1].imshow(img_display, cmap="gray" if color_mode == "GRAY" else None)

        # Draw cropping box
        height, width = img.shape[:2]
        left = crop_params["crop_left"]
        right = width - crop_params["crop_right"]
        top = crop_params["crop_top"]
        bottom = height - crop_params["crop_bottom"]

        import matplotlib.patches as patches

        rect = patches.Rectangle(
            (left, top), right - left, bottom - top, linewidth=2, edgecolor="red", facecolor="none"
        )
        axes[1].add_patch(rect)
        axes[1].set_title("""Cropping area annotation""")
        axes[1].axis("off")

        # Cropped image
        axes[2].imshow(cropped_display, cmap="gray" if color_mode == "GRAY" else None)
        axes[2].set_title(f"""After cropping \n {cropped.shape[1]} x {cropped.shape[0]}""")
        axes[2].axis("off")

        plt.suptitle(f"""DNA origami image crop preview: {os.path.basename(image_path)}""")
        plt.tight_layout()
        plt.show()

        return cropped


def main():
    """Main Function: Command-Line Interface"""
    parser = argparse.ArgumentParser(
        description="""Intelligent Uniform Cropping Tool for DNA Origami Images"""
    )
    parser.add_argument("--input", "-i", required=True, help="""Input Image Directory""")
    parser.add_argument("--output", "-o", required=True, help="""Output directory""")
    parser.add_argument(
        "--sample_size",
        "-s",
        type=int,
        default=10,
        help="""Number of samples used for analysis (default: 10)""",
    )
    parser.add_argument(
        "--preview",
        "-p",
        action="store_true",
        help="""Preview cropping effect (preview before processing)""",
    )
    parser.add_argument(
        "--force", "-f", action="store_true", help="""Force recalculation of cropping parameters"""
    )
    parser.add_argument("--quiet", "-q", action="store_true", help="""Quiet mode, reduced output""")

    args = parser.parse_args()

    # Create Cropper
    cropper = DNACropper(verbose=not args.quiet)

    try:
        # Preview Mode
        if args.preview:
            # Find an image to preview
            input_path = Path(args.input)
            image_files = list(input_path.glob("*.jpg")) + list(input_path.glob("*.png"))
            if image_files:
                preview_image = str(image_files[0])
                print(f"""Preview image: {preview_image}""")

                # Calculate parameters first
                valid_regions = cropper.analyze_sample_images([preview_image], 1)
                crop_params = cropper.calculate_unified_crop_params(valid_regions)

                # Preview
                cropper.preview_crop(preview_image, crop_params)

                response = input("""Do you want to continue with batch cropping? (y/n):""")
                if response.lower() != "y":
                    print("""Cancel operation""")
                    return
            else:
                print("""No image file found for preview""")

        # Perform batch cropping
        crop_params, failed_images = cropper.batch_crop_images(
            args.input, args.output, sample_size=args.sample_size, force_recalculate=args.force
        )

        if failed_images:
            print(
                f"""\n Note: {len(failed_images)} images failed to process. For details, see {args.output} /crop_errors.log"""
            )

        print(
            f"""\n Cropping complete! Parameters have been saved to {args.output} /crop_params.json"""
        )

    except Exception as e:
        print(f"""Error: {e}""")
        sys.exit(1)


if __name__ == "__main__":
    main()
