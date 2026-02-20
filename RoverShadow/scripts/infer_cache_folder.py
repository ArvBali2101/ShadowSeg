import argparse

import sys

from pathlib import Path

import numpy as np


def list_images(img_dir: Path):

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

    return [
        p for p in sorted(img_dir.iterdir()) if p.suffix.lower() in exts and p.is_file()
    ]


def softmax_prob_shadow(seg_logits_chw: np.ndarray) -> np.ndarray:

    seg_logits_chw = seg_logits_chw.astype(np.float32)

    seg_logits_chw = seg_logits_chw - seg_logits_chw.max(axis=0, keepdims=True)

    exp = np.exp(seg_logits_chw)

    denom = exp.sum(axis=0) + 1e-9

    prob = exp[1] / denom

    return prob.astype(np.float32)


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--cfg", required=True)

    ap.add_argument("--ckpt", required=True)

    ap.add_argument("--device", default="cpu")

    ap.add_argument("--img-dir", required=True, help="Folder of input images")

    ap.add_argument("--out-cache", required=True, help="Folder for *.npz prob caches")

    ap.add_argument("--limit", type=int, default=0, help="0 = no limit")

    ap.add_argument("--overwrite", action="store_true")

    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]

    if str(repo_root) not in sys.path:

        sys.path.insert(0, str(repo_root))

    import rovershadow.losses

    from mmengine.config import Config

    from mmseg.apis import init_model, inference_model

    cfg_path = Path(args.cfg).resolve()

    ckpt_path = Path(args.ckpt).resolve()

    img_dir = Path(args.img_dir).resolve()

    out_cache = Path(args.out_cache).resolve()

    out_cache.mkdir(parents=True, exist_ok=True)

    if not img_dir.exists():

        raise FileNotFoundError(f"img-dir not found: {img_dir}")

    imgs = list_images(img_dir)

    if args.limit and args.limit > 0:

        imgs = imgs[: args.limit]

    if not imgs:

        raise RuntimeError(f"No images found in {img_dir}")

    print("[INFO] cfg:", cfg_path)

    print("[INFO] ckpt:", ckpt_path)

    print("[INFO] device:", args.device)

    print("[INFO] img-dir:", img_dir)

    print("[INFO] out-cache:", out_cache)

    print("[INFO] images:", len(imgs))

    cfg = Config.fromfile(str(cfg_path))

    model = init_model(cfg, str(ckpt_path), device=args.device)

    written = 0

    skipped = 0

    for i, img_path in enumerate(imgs, 1):

        stem = img_path.stem

        out_npz = out_cache / f"{stem}.npz"

        if out_npz.exists() and not args.overwrite:

            skipped += 1

            if i % 50 == 0:

                print(f"[SKIP] {i}/{len(imgs)} (skipped so far: {skipped})")

            continue

        result = inference_model(model, str(img_path))

        if not hasattr(result, "seg_logits") or result.seg_logits is None:

            raise RuntimeError(
                "seg_logits missing in inference result. Need mmseg outputs seg_logits."
            )

        seg_logits = result.seg_logits.data.detach().cpu().numpy()

        prob = softmax_prob_shadow(seg_logits)

        np.savez_compressed(out_npz, prob=prob)

        written += 1

        if i % 10 == 0 or i == len(imgs):

            print(
                f"[OK] {i}/{len(imgs)} wrote={written} skipped={skipped} last={out_npz.name}"
            )

    print("[DONE] caching complete")

    print(
        {
            "written": written,
            "skipped": skipped,
            "total": len(imgs),
            "cache_dir": str(out_cache),
        }
    )


if __name__ == "__main__":

    main()
