import argparse, sys

from pathlib import Path

import numpy as np

from PIL import Image


def main():

    p = argparse.ArgumentParser()

    p.add_argument("--cfg", required=True)

    p.add_argument("--ckpt", required=True)

    p.add_argument("--img-dir", required=True)

    p.add_argument("--glob", default="*.png")

    p.add_argument("--out-dir", required=True)

    p.add_argument("--device", default="cpu")

    args = p.parse_args()

    repo_root = Path(__file__).resolve().parents[1]

    sys.path.insert(0, str(repo_root))

    import rovershadow.losses

    from mmengine.config import Config

    from mmseg.apis import init_model, inference_model

    from mmseg.apis.inference import show_result_pyplot

    cfg = Config.fromfile(str(Path(args.cfg).resolve()))

    model = init_model(cfg, str(Path(args.ckpt).resolve()), device=args.device)

    img_dir = Path(args.img_dir).resolve()

    out_dir = Path(args.out_dir).resolve()

    masks_dir = out_dir / "masks"

    overlays_dir = out_dir / "overlays"

    masks_dir.mkdir(parents=True, exist_ok=True)

    overlays_dir.mkdir(parents=True, exist_ok=True)

    imgs = sorted(img_dir.glob(args.glob))

    if not imgs:

        raise RuntimeError("No images found.")

    for im in imgs:

        r = inference_model(model, str(im))

        pred = r.pred_sem_seg.data.squeeze().cpu().numpy().astype(np.uint8)

        mask = (pred > 0).astype(np.uint8) * 255

        mpath = masks_dir / f"{im.stem}_mask.png"

        opath = overlays_dir / f"{im.stem}_overlay.png"

        Image.fromarray(mask).save(mpath)

        show_result_pyplot(
            model, str(im), r, show=False, out_file=str(opath), opacity=0.55
        )

        print("[OK]", im.name)

    print("[DONE] wrote to:", out_dir)


if __name__ == "__main__":

    main()
