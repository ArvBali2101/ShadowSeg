import argparse, sys

from pathlib import Path

import numpy as np

from PIL import Image


def main():

    p = argparse.ArgumentParser()

    p.add_argument("--cfg", required=True)

    p.add_argument("--ckpt", required=True)

    p.add_argument("--img", required=True)

    p.add_argument("--out-overlay", required=True)

    p.add_argument("--out-mask", required=True)

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

    result = inference_model(model, str(Path(args.img).resolve()))

    pred = result.pred_sem_seg.data.squeeze().cpu().numpy().astype(np.uint8)

    mask = (pred > 0).astype(np.uint8) * 255

    out_mask = Path(args.out_mask).resolve()

    out_mask.parent.mkdir(parents=True, exist_ok=True)

    Image.fromarray(mask).save(out_mask)

    out_overlay = Path(args.out_overlay).resolve()

    out_overlay.parent.mkdir(parents=True, exist_ok=True)

    show_result_pyplot(
        model,
        str(Path(args.img).resolve()),
        result,
        show=False,
        out_file=str(out_overlay),
        opacity=0.55,
    )

    print("[OK] mask   :", out_mask)

    print("[OK] overlay:", out_overlay)


if __name__ == "__main__":

    main()
