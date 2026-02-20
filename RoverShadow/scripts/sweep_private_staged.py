import argparse

import sys

from pathlib import Path

import numpy as np

from PIL import Image


def compute_iou_batch(preds01, gts01) -> float:

    tp = fp = fn = 0

    for pm, gm in zip(preds01, gts01):

        pm = pm > 0

        gm = gm > 0

        tp += int(np.logical_and(pm, gm).sum())

        fp += int(np.logical_and(pm, ~gm).sum())

        fn += int(np.logical_and(~pm, gm).sum())

    return tp / (tp + fp + fn + 1e-9)


def load_gt_mask01(path: Path) -> np.ndarray:

    m = np.array(Image.open(path))

    return (m > 0).astype(np.uint8)


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--private-root", required=True)

    ap.add_argument("--cache-dir", default="cache_prob_private")

    ap.add_argument(
        "--thresholds", default="0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9"
    )

    ap.add_argument("--close-ks", default="7")

    ap.add_argument("--min-area", type=int, default=0)

    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]

    if str(repo_root) not in sys.path:

        sys.path.insert(0, str(repo_root))

    from rovershadow.utils.render import prob_to_mask01

    private_root = Path(args.private_root).resolve()

    gt_dir = private_root / "LunarShadowDataset" / "ShadowMasks"

    cache_dir = Path(args.cache_dir).resolve()

    prob_files = sorted(cache_dir.glob("*.npz"))

    probs, gts = [], []

    for pf in prob_files:

        stem = pf.stem

        gt_path = gt_dir / f"{stem}.png"

        if not gt_path.exists():

            continue

        data = np.load(pf)

        prob = data["prob"].astype(np.float32)

        probs.append(prob)

        gts.append(load_gt_mask01(gt_path))

    print(f"[INFO] Loaded {len(probs)} private prob maps")

    thresholds = [float(x) for x in args.thresholds.split(",")]

    close_ks = [int(x) for x in args.close_ks.split(",")]

    best = {"iou": -1.0}

    for thr in thresholds:

        for ck in close_ks:

            preds = []

            for p in probs:

                mask01, _ = prob_to_mask01(
                    p, threshold=thr, adaptive=False, min_area=args.min_area, close_k=ck
                )

                preds.append(mask01)

            iou = compute_iou_batch(preds, gts)

            print(f"thr={thr:.2f}, close_k={ck} -> IoU={iou:.6f}")

            if iou > best["iou"]:

                best = {"iou": float(iou), "threshold": thr, "close_k": ck}

    print("\n[RESULT] BEST PRIVATE STATIC")

    print(best)


if __name__ == "__main__":

    main()
