from PIL import Image

import numpy as np

import os

PRIVATE_MASK_ROOT = r"data/private/LunarShadowDataset/ShadowMasks"


def fix_masks_no_split(mask_root):

    if not os.path.exists(mask_root):

        print(f"Mask folder not found: {mask_root}")

        return

    count = 0

    for filename in os.listdir(mask_root):

        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):

            continue

        path = os.path.join(mask_root, filename)

        mask = Image.open(path).convert("L")

        mask_np = np.array(mask)

        fixed_mask = (mask_np > 0).astype(np.uint8)

        Image.fromarray(fixed_mask).save(path)

        count += 1

    print(f"Fixed {count} private masks in: {mask_root}")

    print("DONE. Private masks are now {0,1}.")


fix_masks_no_split(PRIVATE_MASK_ROOT)
