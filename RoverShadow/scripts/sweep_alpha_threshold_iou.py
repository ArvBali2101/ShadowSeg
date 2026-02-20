import argparse
import glob
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from mmengine.config import Config
from mmseg.apis import inference_model, init_model


def get_shadow_prob_from_logits(
    logits_2chw: torch.Tensor, alpha: float
) -> torch.Tensor:
    """
    logits_2chw: torch tensor shape (2, H, W)
    alpha: logit scaling factor (>1 sharper, <1 softer)
    returns shadow probability map torch shape (H, W) on CPU
    """
    logits = logits_2chw.float()
    scaled = logits * float(alpha)
    probs = F.softmax(scaled, dim=0)
    shadow_prob = probs[1]
    return shadow_prob


def remove_small_components(mask01: np.ndarray, min_area: int) -> np.ndarray:
    """
    mask01: uint8 array values {0,1}
    remove connected components smaller than min_area
    """
    if min_area <= 0:
        return mask01

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask01, connectivity=8
    )
    cleaned = np.zeros_like(mask01)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == i] = 1

    return cleaned


def morph_close(mask01: np.ndarray, k: int) -> np.ndarray:
    """
    morphological closing with kernel kxk (k=0 disables)
    """
    if k <= 0:
        return mask01

    kernel = np.ones((k, k), np.uint8)
    return cv2.morphologyEx(mask01, cv2.MORPH_CLOSE, kernel)


def compute_metrics(pred01: np.ndarray, gt01: np.ndarray):
    """
    both uint8 {0,1}
    returns TP, FP, FN, IoU, F1
    """
    pred = pred01.astype(bool)
    gt = gt01.astype(bool)

    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, np.logical_not(gt)).sum()
    fn = np.logical_and(np.logical_not(pred), gt).sum()

    denom_iou = tp + fp + fn
    iou = (tp / denom_iou) if denom_iou > 0 else 0.0

    denom_f1 = 2 * tp + fp + fn
    f1 = (2 * tp / denom_f1) if denom_f1 > 0 else 0.0

    return int(tp), int(fp), int(fn), float(iou), float(f1)


def load_gt_mask(gt_path: str) -> np.ndarray:
    """
    Loads GT mask and converts to {0,1}.
    Assumes white/255 = shadow, black/0 = background.
    """
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    if gt is None:
        raise FileNotFoundError(f"GT not found: {gt_path}")

    gt01 = (gt >= 128).astype(np.uint8)
    return gt01


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--img-dir", required=True)
    ap.add_argument("--gt-dir", required=True)
    ap.add_argument("--ext", default=".png")
    ap.add_argument("--limit", type=int, default=0)

    ap.add_argument("--tmin", type=float, default=0.5)
    ap.add_argument("--tmax", type=float, default=0.95)
    ap.add_argument("--tstep", type=float, default=0.05)

    ap.add_argument("--alphas", default="1.0")

    ap.add_argument("--min-area", type=int, default=0)
    ap.add_argument("--close-k", type=int, default=0)

    args = ap.parse_args()

    cfg = Config.fromfile(args.cfg)
    model = init_model(cfg, args.ckpt, device=args.device)

    ext = args.ext.lower()
    img_paths = sorted(glob.glob(os.path.join(args.img_dir, f"*{ext}")))
    if not img_paths:
        raise RuntimeError(f"No images found in {args.img_dir} with ext {ext}")

    if args.limit and args.limit > 0:
        img_paths = img_paths[: args.limit]

    alphas = [float(x.strip()) for x in args.alphas.split(",") if x.strip()]

    t = args.tmin
    thresholds = []
    while t <= args.tmax + 1e-9:
        thresholds.append(round(t, 6))
        t += args.tstep

    best = None

    for alpha in alphas:
        for thr in thresholds:
            TP = FP = FN = 0

            for ip in img_paths:
                name = os.path.basename(ip)
                stem = os.path.splitext(name)[0]

                gt_path = os.path.join(args.gt_dir, f"{stem}.png")
                if not os.path.exists(gt_path):
                    gt_path2 = os.path.join(args.gt_dir, f"{stem}{ext}")
                    if os.path.exists(gt_path2):
                        gt_path = gt_path2
                    else:
                        continue

                res = inference_model(model, ip)
                logits = res.seg_logits.data
                shadow_prob = get_shadow_prob_from_logits(logits, alpha=alpha)
                pred01 = (shadow_prob.cpu().numpy() >= thr).astype(np.uint8)

                pred01 = remove_small_components(pred01, min_area=args.min_area)
                pred01 = morph_close(pred01, k=args.close_k)

                gt01 = load_gt_mask(gt_path)
                tp, fp, fn, _, _ = compute_metrics(pred01, gt01)
                TP += tp
                FP += fp
                FN += fn

            denom = TP + FP + FN
            iou = (TP / denom) if denom > 0 else 0.0
            f1 = (2 * TP / (2 * TP + FP + FN)) if (2 * TP + FP + FN) > 0 else 0.0

            record = {
                "alpha": alpha,
                "t": thr,
                "IoU": float(iou),
                "F1": float(f1),
                "TP": int(TP),
                "FP": int(FP),
                "FN": int(FN),
                "images_used": len(img_paths),
                "min_area": args.min_area,
                "close_k": args.close_k,
            }
            print(record)

            if best is None or record["IoU"] > best[0]:
                best = (record["IoU"], record)

    print("\nBEST_BY_IOU:", best[1] if best else None)


if __name__ == "__main__":
    main()
