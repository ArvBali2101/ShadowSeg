import argparse

import sys

from pathlib import Path

import numpy as np

from PIL import Image

def load_image_paths(private_root: Path) -> list[Path]:

    img_dir = private_root / "LunarShadowDataset" / "ShadowImages"

    exts = (".png", ".jpg", ".jpeg", ".bmp")

    paths = [p for p in sorted(img_dir.iterdir()) if p.suffix.lower() in exts]

    return paths

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--cfg", required=True)

    ap.add_argument("--ckpt", required=True)

    ap.add_argument("--device", default="cpu")

    ap.add_argument("--private-root", required=True)

    ap.add_argument("--out-cache", default="cache_prob_private")

    ap.add_argument("--limit", type=int, default=10)

    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]

    if str(repo_root) not in sys.path:

        sys.path.insert(0, str(repo_root))

                                   

    import rovershadow.losses              

    from mmengine.config import Config

    from mmseg.apis import init_model, inference_model

    cfg_path = Path(args.cfg).resolve()

    ckpt_path = Path(args.ckpt).resolve()

    private_root = Path(args.private_root).resolve()

    out_cache = Path(args.out_cache).resolve()

    out_cache.mkdir(parents=True, exist_ok=True)

    cfg = Config.fromfile(str(cfg_path))

    model = init_model(cfg, str(ckpt_path), device=args.device)

    imgs = load_image_paths(private_root)

    if args.limit and args.limit > 0:

        imgs = imgs[: args.limit]

    if not imgs:

        raise RuntimeError(f"No images found under {private_root}")

    print(f"[INFO] Caching {len(imgs)} private images into {out_cache}")

    for i, img_path in enumerate(imgs, 1):

        stem = img_path.stem                 

        out_npz = out_cache / f"{stem}.npz"

        if out_npz.exists():

            continue

        result = inference_model(model, str(img_path))

                                                                                                  

        logits = result.pred_sem_seg.data                                                 

                                                     

        if hasattr(result, "seg_logits") and result.seg_logits is not None:

            seg_logits = result.seg_logits.data           

        else:

                                                   

            data = dict(img_path=str(img_path))

            raise RuntimeError("seg_logits not available in inference result; update inference pipeline to return logits.")

        seg_logits = seg_logits.detach().cpu().numpy().astype(np.float32)

                                  

        seg_logits = seg_logits - seg_logits.max(axis=0, keepdims=True)

        exp = np.exp(seg_logits)

        prob = exp[1] / (exp.sum(axis=0) + 1e-9)                

        np.savez_compressed(out_npz, prob=prob)

        if i % 1 == 0:

            print(f"[OK] {i}/{len(imgs)} wrote {out_npz.name}")

    print("[DONE] caching complete")

if __name__ == "__main__":

    main()
