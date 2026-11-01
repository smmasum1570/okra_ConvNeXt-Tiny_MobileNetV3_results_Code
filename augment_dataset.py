"""
Offline augmentation script for the okra disease image dataset.

Generates Augmented_Image_Dataset from the 300 x 300 original images, as described
in the Data Descriptor "A Field-Based Image Dataset for Automated Recognition of
Okra Plant Diseases Using Computer Vision and Deep Learning".

For every original image, the script:
  1. saves the original under a standardised name, e.g. HL-00001.jpg
  2. saves five augmented variants with a fixed suffix:
       -aug-1  horizontal flip
       -aug-2  rotation by 30 degrees counter-clockwise (size unchanged,
               exposed corners filled with grey RGB 127, 127, 127)
       -aug-3  brightness reduction (factor 0.8)
       -aug-4  contrast enhancement (factor 1.5)
       -aug-5  Gaussian blur (radius 0.8 px)

Images in each class folder are numbered in alphabetical order of their input
file names. Output images are saved with Pillow's default JPEG quality (75).

Usage:
    python augment_dataset.py --input dataset --output augmented_dataset

Requirements: Pillow, tqdm
"""

import argparse
import os

from PIL import Image, ImageEnhance, ImageFilter
from tqdm import tqdm

# Class folder names and the file-name prefix used for each class.
# Expected number of original images per class is given for reference.
FOLDER_PREFIXES = {
    "Fruit_Rot": "FR",                  # 257 images
    "Healthy_Leaf": "HL",               # 1,065 images
    "Insects_Eaten_Leaf": "IEL",        # 745 images
    "Leaf_Curl": "LC",                  # 535 images
    "Leaf_Spot": "LS",                  # 480 images
    "Yellow_Vein_Mosaic_Virus": "YVMV", # 1,185 images
}

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png")


def apply_augmentations(img):
    """Return the five augmented variants of an image, in suffix order (-aug-1 to -aug-5)."""
    return [
        # 1. Horizontal flip (mirror about the vertical axis)
        img.transpose(Image.FLIP_LEFT_RIGHT),
        # 2. Rotation by 30 degrees counter-clockwise (Pillow's positive direction);
        #    image size is kept and the exposed corners are filled with grey
        img.rotate(30, expand=False, fillcolor=(127, 127, 127)),
        # 3. Brightness reduction (factor 0.8)
        ImageEnhance.Brightness(img).enhance(0.8),
        # 4. Contrast enhancement (factor 1.5)
        ImageEnhance.Contrast(img).enhance(1.5),
        # 5. Gaussian blur (radius 0.8 px)
        img.filter(ImageFilter.GaussianBlur(radius=0.8)),
    ]


def process_dataset(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print("Starting renaming and augmentation...\n")

    for folder_name, prefix in FOLDER_PREFIXES.items():
        input_folder = os.path.join(input_dir, folder_name)
        output_folder = os.path.join(output_dir, folder_name)

        if not os.path.exists(input_folder):
            print(f"Warning: folder '{folder_name}' not found in '{input_dir}', skipping.")
            continue
        os.makedirs(output_folder, exist_ok=True)

        # Keep image files only (ignore hidden files) and sort them alphabetically
        image_files = sorted(
            f for f in os.listdir(input_folder)
            if f.lower().endswith(VALID_EXTENSIONS) and not f.startswith(".")
        )
        print(f"Folder: {folder_name} | original images: {len(image_files)} | prefix: {prefix}")

        for idx, file_name in enumerate(tqdm(image_files, desc=f"Processing {prefix}"), start=1):
            img_path = os.path.join(input_folder, file_name)
            try:
                with Image.open(img_path) as img:
                    img = img.convert("RGB")
                    ext = os.path.splitext(file_name)[1].lower() or ".jpg"

                    # Standardised name, e.g. HL-00001
                    base_name = f"{prefix}-{idx:05d}"

                    # Save the original under its new name
                    img.save(os.path.join(output_folder, f"{base_name}{ext}"))

                    # Save the five augmented variants
                    for aug_idx, aug_img in enumerate(apply_augmentations(img), start=1):
                        aug_img.save(os.path.join(output_folder, f"{base_name}-aug-{aug_idx}{ext}"))
            except Exception as e:
                print(f"Error processing {file_name}: {e}")

    print("\nDone: originals renamed and five augmented variants created for each image.")


def main():
    parser = argparse.ArgumentParser(
        description="Generate the offline-augmented okra dataset (five variants per image)."
    )
    parser.add_argument("--input", default="dataset",
                        help="Folder containing the six class sub-folders of original images (default: dataset)")
    parser.add_argument("--output", default="augmented_dataset",
                        help="Folder where renamed originals and augmented images are saved (default: augmented_dataset)")
    args = parser.parse_args()
    process_dataset(args.input, args.output)


if __name__ == "__main__":
    main()
