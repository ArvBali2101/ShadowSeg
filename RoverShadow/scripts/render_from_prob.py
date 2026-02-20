import os, argparse

import numpy as np

import cv2


def connected_components_filter(mask01, min_area):

    if min_area <= 0:
        return mask01

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask01, connectivity=8)

    out = np.zeros_like(mask01, dtype=np.uint8)

    for i in range(1, n):

        if stats[i, cv2.CC_STAT_AREA] >= min_area:

            out[labels == i] = 1

    return out


def morph_close(mask01, k):

    if k <= 0:
        return mask01

    ker = np.ones((k, k), np.uint8)

    return cv2.morphologyEx(mask01, cv2.MORPH_CLOSE, ker)


def overlay_red(image_bgr, mask01, alpha=0.45):

    out = image_bgr.copy()

    red = np.zeros_like(out)
    red[:, :, 2] = 255

    m = mask01.astype(bool)

    out[m] = (out[m] * (1 - alpha) + red[m] * alpha).astype(np.uint8)

    return out


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--prob", required=True)

    ap.add_argument("--img", required=True)

    ap.add_argument("--t", type=float, required=True)

    ap.add_argument("--min-area", type=int, default=0)

    ap.add_argument("--close-k", type=int, default=0)

    ap.add_argument("--out-mask", required=True)

    ap.add_argument("--out-overlay", required=True)

    args = ap.parse_args()

    prob = np.load(args.prob)["prob"]

    mask01 = (prob >= args.t).astype(np.uint8)

    mask01 = connected_components_filter(mask01, args.min_area)

    mask01 = morph_close(mask01, args.close_k)

    os.makedirs(os.path.dirname(args.out_mask), exist_ok=True)

    os.makedirs(os.path.dirname(args.out_overlay), exist_ok=True)

    cv2.imwrite(args.out_mask, (mask01 * 255).astype(np.uint8))

    img = cv2.imread(args.img, cv2.IMREAD_COLOR)

    ov = overlay_red(img, mask01, alpha=0.45)

    cv2.imwrite(args.out_overlay, ov)

    print("[OK] wrote:", args.out_mask)

    print("[OK] wrote:", args.out_overlay)


if __name__ == "__main__":

    main()
