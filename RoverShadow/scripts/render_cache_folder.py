import argparse

import sys

from pathlib import Path

import numpy as np

from PIL import Image


def list_npz(cache_dir: Path):

    return sorted(cache_dir.glob("*.npz"))


def load_image_any(path: Path) -> Image.Image:

    return Image.open(path).convert("RGB")


def make_overlay(
    img_rgb: Image.Image, mask01: np.ndarray, opacity: float = 0.55
) -> Image.Image:

    arr = np.array(img_rgb).astype(np.float32)

    h, w = mask01.shape

    if arr.shape[0] != h or arr.shape[1] != w:

        img_rgb = img_rgb.resize((w, h), resample=Image.BILINEAR)

        arr = np.array(img_rgb).astype(np.float32)

    red = np.zeros_like(arr)

    red[..., 0] = 255.0

    m = mask01.astype(bool)

    out = arr.copy()

    out[m] = (1.0 - opacity) * out[m] + opacity * red[m]

    out = np.clip(out, 0, 255).astype(np.uint8)

    return Image.fromarray(out)


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--cache-dir", required=True, help="Folder with *.npz prob files")

    ap.add_argument(
        "--img-dir", required=True, help="Folder with original images (same stems)"
    )

    ap.add_argument("--out-masks", required=True)

    ap.add_argument("--out-overlays", required=True)

    ap.add_argument("--threshold", type=float, required=True)

    ap.add_argument("--min-area", type=int, default=0)

    ap.add_argument("--close-k", type=int, default=0)

    ap.add_argument("--opacity", type=float, default=0.55)

    ap.add_argument("--limit", type=int, default=0)

    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]

    if str(repo_root) not in sys.path:

        sys.path.insert(0, str(repo_root))

    from rovershadow.utils.render import prob_to_mask01

    cache_dir = Path(args.cache_dir).resolve()

    img_dir = Path(args.img_dir).resolve()

    out_masks = Path(args.out_masks).resolve()

    out_overlays = Path(args.out_overlays).resolve()

    out_masks.mkdir(parents=True, exist_ok=True)

    out_overlays.mkdir(parents=True, exist_ok=True)

    prob_files = list_npz(cache_dir)

    if args.limit and args.limit > 0:

        prob_files = prob_files[: args.limit]

    if not prob_files:

        raise RuntimeError(f"No .npz in {cache_dir}")

    print("[INFO] cache-dir:", cache_dir)

    print("[INFO] img-dir:", img_dir)

    print("[INFO] out-masks:", out_masks)

    print("[INFO] out-overlays:", out_overlays)

    print("[INFO] N:", len(prob_files))

    print("[INFO] thr/min_area/close_k:", args.threshold, args.min_area, args.close_k)

    done = 0

    for i, pf in enumerate(prob_files, 1):

        stem = pf.stem

        img_path = None

        for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:

            cand = img_dir / f"{stem}{ext}"

            if cand.exists():

                img_path = cand

                break

        if img_path is None:

            continue

        data = np.load(pf)

        prob = data["prob"].astype(np.float32)

        mask01 = prob_to_mask01(
            prob, threshold=args.threshold, min_area=args.min_area, close_k=args.close_k
        )

        mask255 = (mask01 * 255).astype(np.uint8)

        Image.fromarray(mask255).save(out_masks / f"{stem}.png")

        img = load_image_any(img_path)

        overlay = make_overlay(img, mask01, opacity=float(args.opacity))

        overlay.save(out_overlays / f"{stem}.png")

        done += 1

        if i % 50 == 0 or i == len(prob_files):

            print(f"[OK] {i}/{len(prob_files)} done={done} last={stem}")

    print("[DONE] render complete")

    print({"rendered": done, "total_npz": len(prob_files)})


if __name__ == "__main__":

    main()
