from PIL import Image

import numpy as np

import os

PUBLIC_MASK_ROOT = r"data/public/Rover_Shadow_Public_Dataset/ShadowMasks"

PRIVATE_MASK_ROOT = r"data/private/LunarShadowDataset/ShadowMasks"


def fix_masks(mask_root):

    for split in ["train", "val"]:

        folder = os.path.join(mask_root, split)

        if not os.path.exists(folder):

            print(f"Skipping missing folder: {folder}")

            continue

        for filename in os.listdir(folder):

            if not filename.lower().endswith((".png", ".jpg")):

                continue

            path = os.path.join(folder, filename)

            mask = Image.open(path).convert("L")

            mask_np = np.array(mask)

            fixed_mask = (mask_np > 0).astype(np.uint8)

            Image.fromarray(fixed_mask).save(path)

        print(f"Fixed masks in: {folder}")


print("Fixing PUBLIC dataset masks...")

fix_masks(PUBLIC_MASK_ROOT)

print("Fixing PRIVATE dataset masks...")

fix_masks(PRIVATE_MASK_ROOT)

print("DONE. All masks are now {0,1}.")
