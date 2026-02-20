import argparse, sys, os

from pathlib import Path

import numpy as np

from PIL import Image


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--pred-dir", required=True)

    ap.add_argument("--gt-dir", required=True)

    ap.add_argument("--suffix", default="_mask.png")

    args = ap.parse_args()

    pred_dir = Path(args.pred_dir)

    gt_dir = Path(args.gt_dir)

    preds = sorted(pred_dir.glob(f"*{args.suffix}"))

    if not preds:

        raise RuntimeError(f"No preds found in {pred_dir}")

    tp = fp = fn = 0

    used = 0

    for p in preds:

        base = p.name.replace(args.suffix, "") + ".png"

        gt = gt_dir / base

        if not gt.exists():

            continue

        pm = np.array(Image.open(p)) > 0

        gm = np.array(Image.open(gt)) > 0

        tp += int(np.logical_and(pm, gm).sum())

        fp += int(np.logical_and(pm, ~gm).sum())

        fn += int(np.logical_and(~pm, gm).sum())

        used += 1

    iou = tp / (tp + fp + fn + 1e-9)

    f1 = 2 * tp / (2 * tp + fp + fn + 1e-9)

    print(
        {
            "images_used": used,
            "IoU_shadow": float(iou),
            "F1_shadow": float(f1),
            "TP": tp,
            "FP": fp,
            "FN": fn,
        }
    )


if __name__ == "__main__":

    main()
