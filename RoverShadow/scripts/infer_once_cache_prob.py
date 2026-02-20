import argparse

import sys

from pathlib import Path

import numpy as np


def softmax_prob_class1(seg_logits: np.ndarray) -> np.ndarray:
    """
    seg_logits: [C,H,W] float32
    returns prob for class 1: [H,W] float32
    """

    seg_logits = seg_logits.astype(np.float32)

    seg_logits = seg_logits - seg_logits.max(axis=0, keepdims=True)

    exp = np.exp(seg_logits)

    denom = exp.sum(axis=0) + 1e-9

    prob1 = exp[1] / denom

    return prob1.astype(np.float32)


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--cfg", required=True)

    ap.add_argument("--ckpt", required=True)

    ap.add_argument("--img", required=True)

    ap.add_argument("--out-npz", required=True)

    ap.add_argument("--device", default="cpu")

    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]

    if str(repo_root) not in sys.path:

        sys.path.insert(0, str(repo_root))

    import rovershadow.losses

    from mmengine.config import Config

    from mmseg.apis import init_model, inference_model

    cfg_path = Path(args.cfg).resolve()

    ckpt_path = Path(args.ckpt).resolve()

    img_path = Path(args.img).resolve()

    out_npz = Path(args.out_npz).resolve()

    out_npz.parent.mkdir(parents=True, exist_ok=True)

    print("[INFO] cfg:", cfg_path)

    print("[INFO] ckpt:", ckpt_path)

    print("[INFO] img:", img_path)

    print("[INFO] out_npz:", out_npz)

    print("[INFO] device:", args.device)

    cfg = Config.fromfile(str(cfg_path))

    model = init_model(cfg, str(ckpt_path), device=args.device)

    result = inference_model(model, str(img_path))

    if hasattr(result, "seg_logits") and result.seg_logits is not None:

        seg_logits = result.seg_logits.data.detach().cpu().numpy()

    else:

        raise RuntimeError(
            "seg_logits not found in inference result. Need seg_logits to compute prob map."
        )

    if seg_logits.ndim != 3 or seg_logits.shape[0] < 2:

        raise RuntimeError(
            f"Unexpected seg_logits shape: {seg_logits.shape} (expected [C,H,W] with C>=2)"
        )

    prob = softmax_prob_class1(seg_logits)

    np.savez_compressed(out_npz, prob=prob)

    print("[OK] wrote prob:", out_npz)


if __name__ == "__main__":

    main()
