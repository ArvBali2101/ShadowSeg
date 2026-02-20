import argparse

import sys

from pathlib import Path

import numpy as np

try:

    import cv2

except Exception:

    cv2 = None


def _require_cv2():

    if cv2 is None:

        raise ImportError(
            "OpenCV is required here. Install with: pip install opencv-python"
        )


def _read_bgr(img_path: Path) -> np.ndarray:

    _require_cv2()

    bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)

    if bgr is None:

        raise RuntimeError(f"Failed to read image: {img_path}")

    return bgr


def _write_png(path: Path, img: np.ndarray) -> None:

    _require_cv2()

    path.parent.mkdir(parents=True, exist_ok=True)

    ok = cv2.imwrite(str(path), img)

    if not ok:

        raise RuntimeError(f"Failed to write: {path}")


def _extract_shadow_prob(result) -> np.ndarray | None:
    """
    Try to extract per-pixel shadow probability map HxW float32 in [0,1].

    Works if mmseg returns logits in the SegDataSample.
    If logits are not present, returns None.
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

    shadow_prob = prob[1].detach().cpu().numpy().astype(np.float32)

    return shadow_prob


def _overlay_mask(
    bgr: np.ndarray, mask01: np.ndarray, alpha: float = 0.55
) -> np.ndarray:
    """
    Simple overlay: tint shadow pixels red-ish by boosting R channel.
    mask01: HxW {0,1}
    """

    _require_cv2()

    out = bgr.copy()

    m = mask01.astype(np.uint8) > 0

    overlay = out.copy()

    overlay[m, 2] = 255

    out[m] = (alpha * overlay[m] + (1.0 - alpha) * out[m]).astype(np.uint8)

    return out


def main():

    p = argparse.ArgumentParser()

    p.add_argument("--cfg", required=True)

    p.add_argument("--ckpt", required=True)

    p.add_argument("--img", required=True)

    p.add_argument("--out", required=True)

    p.add_argument("--device", default="cpu")

    p.add_argument("--threshold", type=float, default=0.5)

    p.add_argument("--min-area", type=int, default=0)

    p.add_argument("--close-k", type=int, default=0)

    p.add_argument("--overlay-alpha", type=float, default=0.55)

    p.add_argument(
        "--save-prob", action="store_true", help="Save prob map as *_prob.npz"
    )

    p.add_argument("--out-mask", default=None, help="Explicit path for mask PNG")

    p.add_argument("--out-prob", default=None, help="Explicit path for prob NPZ")

    args = p.parse_args()

    repo_root = Path(__file__).resolve().parents[1]

    if str(repo_root) not in sys.path:

        sys.path.insert(0, str(repo_root))

    import rovershadow.losses

    from mmengine.config import Config

    from mmseg.apis import init_model, inference_model

    from rovershadow.utils.render import prob_to_mask01

    cfg_path = Path(args.cfg).resolve()

    ckpt_path = Path(args.ckpt).resolve()

    img_path = Path(args.img).resolve()

    out_overlay_path = Path(args.out).resolve()

    if args.out_mask is None:

        out_mask_path = out_overlay_path.with_name(out_overlay_path.stem + "_mask.png")

    else:

        out_mask_path = Path(args.out_mask).resolve()

    if args.out_prob is None:

        out_prob_path = out_overlay_path.with_name(out_overlay_path.stem + "_prob.npz")

    else:

        out_prob_path = Path(args.out_prob).resolve()

    print("[INFO] cfg:", cfg_path)

    print("[INFO] ckpt:", ckpt_path)

    print("[INFO] img:", img_path)

    print("[INFO] out_overlay:", out_overlay_path)

    print("[INFO] out_mask:", out_mask_path)

    print("[INFO] out_prob:", out_prob_path)

    print("[INFO] device:", args.device)

    print(
        "[INFO] threshold/min_area/close_k:",
        args.threshold,
        args.min_area,
        args.close_k,
    )

    cfg = Config.fromfile(str(cfg_path))

    model = init_model(cfg, str(ckpt_path), device=args.device)

    result = inference_model(model, str(img_path))

    shadow_prob = _extract_shadow_prob(result)

    if shadow_prob is None:

        print(
            "[WARN] No logits/prob found in inference result. Falling back to pred_sem_seg mask."
        )

        if not hasattr(result, "pred_sem_seg") or result.pred_sem_seg is None:

            raise RuntimeError(
                "Inference result has no pred_sem_seg either. Cannot produce output."
            )

        pred = result.pred_sem_seg

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

        mask01 = (pred.astype(np.uint8) > 0).astype(np.uint8)

        from rovershadow.utils.postproc import apply_postproc

        mask01 = apply_postproc(
            mask01, min_area=int(args.min_area), close_k=int(args.close_k)
        )

    else:

        mask01 = prob_to_mask01(
            shadow_prob,
            threshold=float(args.threshold),
            min_area=int(args.min_area),
            close_k=int(args.close_k),
        )

        if args.save_prob:

            out_prob_path.parent.mkdir(parents=True, exist_ok=True)

            np.savez_compressed(out_prob_path, prob=shadow_prob.astype(np.float16))

            print("[OK] wrote prob:", out_prob_path)

    bgr = _read_bgr(img_path)

    mask_img = mask01.astype(np.uint8) * 255

    overlay = _overlay_mask(bgr, mask01, alpha=float(args.overlay_alpha))

    _write_png(out_mask_path, mask_img)

    _write_png(out_overlay_path, overlay)

    print("[OK] wrote mask:", out_mask_path)

    print("[OK] wrote overlay:", out_overlay_path)


if __name__ == "__main__":

    main()
