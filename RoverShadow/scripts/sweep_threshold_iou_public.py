import argparse

import sys

from pathlib import Path

import numpy as np

from PIL import Image


def _extract_shadow_prob(result):
    """Return HxW float32 shadow prob in [0,1], or None if logits not available."""

    import torch

    import torch.nn.functional as F

    logits = None

    for attr in ("seg_logits", "logits", "pred_logits"):

        if hasattr(result, attr):

            logits = getattr(result, attr)

            if logits is not None:

                break

    if logits is None:

        return None

    if hasattr(logits, "data"):

        logits = logits.data

    if not torch.is_tensor(logits):

        return None

    if logits.ndim == 4:

        logits = logits[0]

    if logits.ndim != 3:

        return None

    prob = F.softmax(logits.float(), dim=0)

    if prob.shape[0] < 2:

        return None

    return prob[1].detach().cpu().numpy().astype(np.float32)


def _load_gt_mask01(gt_path: Path) -> np.ndarray:

    m = np.array(Image.open(gt_path))

    return (m > 0).astype(np.uint8)


def _iou(mask01: np.ndarray, gt01: np.ndarray) -> float:

    inter = np.logical_and(mask01 == 1, gt01 == 1).sum()

    union = np.logical_or(mask01 == 1, gt01 == 1).sum()

    return float(inter) / float(union + 1e-9)


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--cfg", required=True)

    ap.add_argument("--ckpt", required=True)

    ap.add_argument("--device", default="cpu")

    ap.add_argument(
        "--public-root",
        default="data/public/Rover_Shadow_Public_Dataset",
        help="Path to Rover_Shadow_Public_Dataset",
    )

    ap.add_argument("--limit", type=int, default=50)

    ap.add_argument("--thr-min", type=float, default=0.30)

    ap.add_argument("--thr-max", type=float, default=0.70)

    ap.add_argument("--thr-step", type=float, default=0.05)

    ap.add_argument("--min-areas", default="0,50,100,200,400,800")

    ap.add_argument("--close-ks", default="0,3,5,7")

    ap.add_argument("--cache-dir", default="cache_prob_public_val")

    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]

    if str(repo_root) not in sys.path:

        sys.path.insert(0, str(repo_root))

    import rovershadow.losses

    from rovershadow.utils.render import prob_to_mask01

    from mmengine.config import Config

    from mmseg.apis import init_model, inference_model

    public_root = Path(args.public_root).resolve()

    img_dir = public_root / "ShadowImages" / "val"

    gt_dir = public_root / "ShadowMasks" / "val"

    if not img_dir.exists():

        raise RuntimeError(f"Missing image dir: {img_dir}")

    if not gt_dir.exists():

        raise RuntimeError(f"Missing gt dir: {gt_dir}")

    imgs = sorted(
        list(img_dir.glob("*.jpg"))
        + list(img_dir.glob("*.png"))
        + list(img_dir.glob("*.jpeg"))
    )

    if not imgs:

        raise RuntimeError(f"No images found in: {img_dir}")

    if args.limit and args.limit > 0:

        imgs = imgs[: args.limit]

    cache_dir = Path(args.cache_dir).resolve()

    cache_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config.fromfile(str(Path(args.cfg).resolve()))

    model = init_model(cfg, str(Path(args.ckpt).resolve()), device=args.device)

    thresholds = (
        np.arange(args.thr_min, args.thr_max + 1e-9, args.thr_step).round(4).tolist()
    )

    min_areas = [int(x) for x in args.min_areas.split(",") if x.strip() != ""]

    close_ks = [int(x) for x in args.close_ks.split(",") if x.strip() != ""]

    probs = []

    gts = []

    used = 0

    for img_path in imgs:

        stem = img_path.stem

        gt_path = gt_dir / f"{stem}.png"

        if not gt_path.exists():

            alt = gt_dir / f"{stem}.jpg"

            if alt.exists():

                gt_path = alt

            else:

                continue

        cache_path = cache_dir / f"{stem}.npz"

        if cache_path.exists():

            prob = np.load(cache_path)["prob"].astype(np.float32)

        else:

            result = inference_model(model, str(img_path))

            prob = _extract_shadow_prob(result)

            if prob is None:

                raise RuntimeError(
                    "Could not extract seg_logits/prob from mmseg result. "
                    "Your model result must expose logits for slider-style thresholding."
                )

            np.savez_compressed(cache_path, prob=prob.astype(np.float16))

        gt01 = _load_gt_mask01(gt_path)

        probs.append(prob)

        gts.append(gt01)

        used += 1

    if used == 0:

        raise RuntimeError("No image/gt pairs were usable.")

    print(f"[INFO] Loaded {used} prob maps + GT masks (cache: {cache_dir})")

    print(
        f"[INFO] Grid: thresholds={len(thresholds)}, min_areas={min_areas}, close_ks={close_ks}"
    )

    best = {"iou": -1.0, "threshold": None, "min_area": None, "close_k": None}

    for thr in thresholds:

        for min_area in min_areas:

            for close_k in close_ks:

                ious = []

                for prob, gt01 in zip(probs, gts):

                    mask01 = prob_to_mask01(
                        prob, threshold=thr, min_area=min_area, close_k=close_k
                    )

                    ious.append(_iou(mask01, gt01))

                mean_iou = float(np.mean(ious))

                if mean_iou > best["iou"]:

                    best = {
                        "iou": mean_iou,
                        "threshold": thr,
                        "min_area": min_area,
                        "close_k": close_k,
                    }

                    print("[BEST]", best)

    print("\n[RESULT] Best params (public val subset/full):")

    print(best)


if __name__ == "__main__":

    main()
