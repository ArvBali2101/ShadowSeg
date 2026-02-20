import argparse

import sys

from pathlib import Path

import numpy as np

from PIL import Image


def _load_gt_mask01(gt_path: Path) -> np.ndarray:

    m = np.array(Image.open(gt_path))

    return (m > 0).astype(np.uint8)


def _iou(mask01: np.ndarray, gt01: np.ndarray) -> float:

    inter = np.logical_and(mask01 == 1, gt01 == 1).sum()

    union = np.logical_or(mask01 == 1, gt01 == 1).sum()

    return float(inter) / float(union + 1e-9)


def _extract_pred_mask01(result) -> np.ndarray:
    """Extract mmseg predicted hard mask (HxW uint8 {0,1})."""

    pred = getattr(result, "pred_sem_seg", None)

    if pred is None:

        raise RuntimeError("result.pred_sem_seg missing")

    if hasattr(pred, "data"):

        pred = pred.data

    try:

        import torch

        if torch.is_tensor(pred):

            if pred.ndim == 3:

                pred = pred[0]

            pred = pred.detach().cpu().numpy()

    except Exception:

        pass

    pred = pred.astype(np.uint8)

    return (pred > 0).astype(np.uint8)


def _extract_shadow_prob(result) -> np.ndarray | None:
    """
    Try to extract per-pixel shadow probability map HxW float32 in [0,1].
    Returns None if logits not available.
    """

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


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--cfg", required=True)

    ap.add_argument("--ckpt", required=True)

    ap.add_argument("--device", default="cpu")

    ap.add_argument("--public-root", default="data/public/Rover_Shadow_Public_Dataset")

    ap.add_argument("--n", type=int, default=10)

    ap.add_argument("--threshold", type=float, default=0.5)

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

    imgs = sorted(
        list(img_dir.glob("*.jpg"))
        + list(img_dir.glob("*.png"))
        + list(img_dir.glob("*.jpeg"))
    )

    imgs = imgs[: args.n]

    cfg = Config.fromfile(str(Path(args.cfg).resolve()))

    model = init_model(cfg, str(Path(args.ckpt).resolve()), device=args.device)

    ious_pred = []

    ious_prob = []

    ok_prob = 0

    for img_path in imgs:

        stem = img_path.stem

        gt_path = gt_dir / f"{stem}.png"

        if not gt_path.exists():

            print("[SKIP] missing gt:", gt_path)

            continue

        gt01 = _load_gt_mask01(gt_path)

        result = inference_model(model, str(img_path))

        pred01 = _extract_pred_mask01(result)

        iou_pred = _iou(pred01, gt01)

        ious_pred.append(iou_pred)

        prob = _extract_shadow_prob(result)

        if prob is None:

            print("[WARN] no logits for prob:", img_path.name)

        else:

            mask01 = prob_to_mask01(
                prob, threshold=float(args.threshold), min_area=0, close_k=0
            )

            iou_prob = _iou(mask01, gt01)

            ious_prob.append(iou_prob)

            ok_prob += 1

        print(
            f"{img_path.name} | IoU(pred_sem_seg)={iou_pred:.4f} | "
            f"IoU(prob>=thr)={'%.4f' % ious_prob[-1] if ok_prob else 'NA'}"
        )

    print("\n[SUMMARY]")

    print({"n_images": len(imgs), "prob_available": ok_prob})

    print({"mean_iou_pred_sem_seg": float(np.mean(ious_pred)) if ious_pred else None})

    print({"mean_iou_prob_threshold": float(np.mean(ious_prob)) if ious_prob else None})


if __name__ == "__main__":

    main()
