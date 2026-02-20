import argparse

import sys

from pathlib import Path

import numpy as np

from PIL import Image


def compute_iou_batch(preds01: list[np.ndarray], gts01: list[np.ndarray]) -> float:

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

    ap.add_argument("--public-root", required=True)

    ap.add_argument("--cache-dir", default="cache_prob_public_val")

    ap.add_argument("--thresholds", default="0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6")

    ap.add_argument("--close-ks", default="0,3,5,7")

    ap.add_argument("--min-areas", default="0,50,100,200,400,800")

    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]

    if str(repo_root) not in sys.path:

        sys.path.insert(0, str(repo_root))

    from rovershadow.utils.render import prob_to_mask01

    public_root = Path(args.public_root).resolve()

    img_dir = public_root / "ShadowImages" / "val"

    gt_dir = public_root / "ShadowMasks" / "val"

    cache_dir = Path(args.cache_dir).resolve()

    if not cache_dir.exists():

        raise RuntimeError(
            f"Cache dir not found: {cache_dir}\n"
            f"Expected .npz prob maps already cached here."
        )

    prob_files = sorted(cache_dir.glob("*.npz"))

    if not prob_files:

        raise RuntimeError(f"No .npz files in cache: {cache_dir}")

    probs = []

    gts = []

    names = []

    for pf in prob_files:

        stem = pf.stem

        gt_path = gt_dir / f"{stem}.png"

        if not gt_path.exists():

            continue

        data = np.load(pf)

        if "prob" not in data:

            raise RuntimeError(f"{pf} missing 'prob' key")

        prob = data["prob"].astype(np.float32)

        probs.append(prob)

        gts.append(load_gt_mask01(gt_path))

        names.append(stem)

    print(f"[INFO] Loaded cached probs+GT: {len(probs)} (cache={cache_dir})")

    thresholds = [float(x) for x in args.thresholds.split(",")]

    close_ks = [int(x) for x in args.close_ks.split(",")]

    min_areas = [int(x) for x in args.min_areas.split(",")]

    bestA = {"iou": -1.0, "threshold": None, "close_k": None}

    for thr in thresholds:

        for ck in close_ks:

            preds = [
                prob_to_mask01(p, threshold=thr, min_area=0, close_k=ck) for p in probs
            ]

            iou = compute_iou_batch(preds, gts)

            if iou > bestA["iou"]:

                bestA = {"iou": float(iou), "threshold": float(thr), "close_k": int(ck)}

                print("[BEST A]", bestA)

    bestB = {
        "iou": bestA["iou"],
        "threshold": bestA["threshold"],
        "close_k": bestA["close_k"],
        "min_area": 0,
    }

    thr = float(bestA["threshold"])

    ck = int(bestA["close_k"])

    for ma in min_areas:

        preds = [
            prob_to_mask01(p, threshold=thr, min_area=ma, close_k=ck) for p in probs
        ]

        iou = compute_iou_batch(preds, gts)

        if iou > bestB["iou"]:

            bestB = {
                "iou": float(iou),
                "threshold": thr,
                "close_k": ck,
                "min_area": int(ma),
            }

            print("[BEST B]", bestB)

    print("\n[RESULT] Best params (public val, staged):")

    print(bestB)


if __name__ == "__main__":

    main()
