import argparse

import os

from pathlib import Path

import numpy as np

from PIL import Image

import torch

import torch.nn.functional as F

from mmengine.config import Config

from mmseg.apis import init_model, inference_model


def parse_args():

    p = argparse.ArgumentParser()

    p.add_argument("--cfg", required=True)

    p.add_argument("--ckpt", required=True)

    p.add_argument("--img-dir", required=True)

    p.add_argument("--gt-dir", required=True)

    p.add_argument(
        "--ext", default=".png", help="image extension filter, e.g. .png or .jpg"
    )

    p.add_argument("--device", default="cpu")

    p.add_argument("--tmin", type=float, default=0.30)

    p.add_argument("--tmax", type=float, default=0.90)

    p.add_argument("--tstep", type=float, default=0.05)

    p.add_argument("--limit", type=int, default=0, help="0 = all, else first N images")

    return p.parse_args()


def binarize_gt(arr):

    if arr.ndim == 3:

        arr = arr[..., 0]

    return (arr > 0).astype(np.uint8)


def main():

    a = parse_args()

    cfg = Config.fromfile(a.cfg)

    model = init_model(cfg, a.ckpt, device=a.device)

    img_dir = Path(a.img_dir)

    gt_dir = Path(a.gt_dir)

    names = sorted(
        [
            p.name
            for p in img_dir.iterdir()
            if p.is_file() and p.suffix.lower() == a.ext.lower()
        ]
    )

    if a.limit and a.limit > 0:

        names = names[: a.limit]

    if not names:

        raise RuntimeError(f"No images found in {img_dir} with ext {a.ext}")

    thresholds = np.arange(a.tmin, a.tmax + 1e-9, a.tstep).tolist()

    best = None

    for t in thresholds:

        tp = fp = fn = 0

        for n in names:

            img_path = img_dir / n

            gt_path = gt_dir / n

            if not gt_path.exists():

                continue

            res = inference_model(model, str(img_path))

            logits = res.seg_logits.data

            probs = F.softmax(logits, dim=0)

            p_shadow = probs[1].detach().cpu().numpy()

            pred = (p_shadow >= t).astype(np.uint8)

            gt = np.array(Image.open(gt_path))

            gt = binarize_gt(gt)

            tp += int(((pred == 1) & (gt == 1)).sum())

            fp += int(((pred == 1) & (gt == 0)).sum())

            fn += int(((pred == 0) & (gt == 1)).sum())

        iou = tp / (tp + fp + fn + 1e-9)

        f1 = 2 * tp / (2 * tp + fp + fn + 1e-9)

        row = {
            "t": round(t, 3),
            "IoU": float(iou),
            "F1": float(f1),
            "TP": int(tp),
            "FP": int(fp),
            "FN": int(fn),
        }

        print(row)

        if best is None or row["IoU"] > best["IoU"]:

            best = row

    print("\nBEST_BY_IOU:", best)


if __name__ == "__main__":

    main()
