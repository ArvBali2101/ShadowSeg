import argparse

from pathlib import Path

import numpy as np

from PIL import Image


def make_overlay_rgb(
    image_rgb: np.ndarray, mask01: np.ndarray, alpha: float = 0.55
) -> np.ndarray:
    """
    image_rgb: HxWx3 uint8
    mask01: HxW uint8 {0,1}
    returns: HxWx3 uint8 overlay (red tint on mask)
    """

    img = image_rgb.astype(np.float32)

    m = (mask01 > 0).astype(np.float32)[..., None]

    red = np.zeros_like(img)

    red[..., 0] = 255.0

    out = img * (1.0 - alpha * m) + red * (alpha * m)

    out = np.clip(out, 0, 255).astype(np.uint8)

    return out


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--prob-npz", required=True)

    ap.add_argument("--img", required=True, help="Original image path (for overlay).")

    ap.add_argument(
        "--out-prefix", required=True, help="Output prefix path without extension."
    )

    ap.add_argument("--threshold", type=float, required=True)

    ap.add_argument("--min-area", type=int, default=0)

    ap.add_argument("--close-k", type=int, default=0)

    ap.add_argument("--alpha", type=float, default=0.55, help="Overlay alpha")

    args = ap.parse_args()

    from rovershadow.utils.render import prob_to_mask01

    prob_npz = Path(args.prob_npz).resolve()

    img_path = Path(args.img).resolve()

    out_prefix = Path(args.out_prefix).resolve()

    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    data = np.load(prob_npz)

    prob = data["prob"].astype(np.float32)

    mask01 = prob_to_mask01(
        prob, threshold=args.threshold, min_area=args.min_area, close_k=args.close_k
    )

    out_mask = Path(str(out_prefix) + "_mask.png")

    Image.fromarray((mask01 * 255).astype(np.uint8)).save(out_mask)

    img = Image.open(img_path).convert("RGB")

    img_rgb = np.array(img)

    overlay = make_overlay_rgb(img_rgb, mask01, alpha=float(args.alpha))

    out_overlay = Path(str(out_prefix) + "_overlay.png")

    Image.fromarray(overlay).save(out_overlay)

    print("[OK] wrote mask:", out_mask)

    print("[OK] wrote overlay:", out_overlay)


if __name__ == "__main__":

    main()
