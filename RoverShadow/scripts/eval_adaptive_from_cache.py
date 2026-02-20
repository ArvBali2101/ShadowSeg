import argparse

from pathlib import Path

import numpy as np

from PIL import Image

from rovershadow.utils.render import prob_to_mask01


def load_prob_npz(p: Path) -> np.ndarray:

    d = np.load(str(p))

    for k in ("prob", "shadow_prob", "p"):

        if k in d:

            return d[k].astype(np.float32)

    if len(d.files) == 1:

        return d[d.files[0]].astype(np.float32)

    raise KeyError(f"Could not find prob array in {p.name}. Keys={d.files}")


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--cache-dir", required=True, help="folder of *.npz prob maps")

    ap.add_argument("--gt-dir", required=True, help="folder of GT masks (png)")

    ap.add_argument("--domain", default="auto", choices=["auto", "public", "private"])

    ap.add_argument("--adaptive", action="store_true", help="use adaptive thresholding")

    ap.add_argument(
        "--threshold", type=float, default=None, help="static threshold if not adaptive"
    )

    ap.add_argument(
        "--target-frac",
        type=float,
        default=None,
        help="override expected shadow area fraction",
    )

    ap.add_argument("--min-area", type=int, default=0)

    ap.add_argument("--close-k", type=int, default=0)

    ap.add_argument("--limit", type=int, default=0)

    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)

    gt_dir = Path(args.gt_dir)

    probs = sorted(cache_dir.glob("*.npz"))

    if args.limit and args.limit > 0:

        probs = probs[: args.limit]

    if not probs:

        raise RuntimeError(f"No .npz files found in {cache_dir}")

    tp = fp = fn = 0

    used = 0

    thr_used_list = []

    for npz_path in probs:

        stem = npz_path.stem

        gt_path = gt_dir / f"{stem}.png"

        if not gt_path.exists():

            continue

        prob = load_prob_npz(npz_path)

        gt = (np.array(Image.open(gt_path)) > 0).astype(np.uint8)

        mask01, used_thr = prob_to_mask01(
            prob,
            threshold=args.threshold,
            adaptive=bool(args.adaptive),
            domain=args.domain,
            target_frac=args.target_frac,
            min_area=args.min_area,
            close_k=args.close_k,
        )

        pred = mask01.astype(bool)

        gm = gt.astype(bool)

        tp += int(np.logical_and(pred, gm).sum())

        fp += int(np.logical_and(pred, ~gm).sum())

        fn += int(np.logical_and(~pred, gm).sum())

        used += 1

        thr_used_list.append(float(used_thr))

    iou = tp / (tp + fp + fn + 1e-9)

    f1 = 2 * tp / (2 * tp + fp + fn + 1e-9)

    out = {
        "images_used": used,
        "IoU_shadow": float(iou),
        "F1_shadow": float(f1),
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
    }

    if thr_used_list:

        out["thr_mean"] = float(np.mean(thr_used_list))

        out["thr_min"] = float(np.min(thr_used_list))

        out["thr_max"] = float(np.max(thr_used_list))

    print(out)


if __name__ == "__main__":

    main()
