import os, glob, json, time, argparse, itertools

import numpy as np

import cv2

import torch

import torch.nn.functional as F

from mmengine.config import Config

from mmseg.apis import init_model, inference_model


def ensure_dir(p):

    os.makedirs(p, exist_ok=True)


def list_images(img_dir, ext, limit):

    paths = sorted(glob.glob(os.path.join(img_dir, f"*{ext}")))

    if not paths:

        raise RuntimeError(f"No images found in {img_dir} with ext={ext}")

    if limit and limit > 0:

        paths = paths[:limit]

    return paths


def gt_for_image(gt_dir, img_path, prefer_ext=".png"):

    stem = os.path.splitext(os.path.basename(img_path))[0]

    p1 = os.path.join(gt_dir, stem + prefer_ext)

    if os.path.exists(p1):

        return p1

    p2 = os.path.join(gt_dir, stem + os.path.splitext(img_path)[1])

    if os.path.exists(p2):

        return p2

    return None


def load_gt_mask01(path):

    gt = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if gt is None:

        raise FileNotFoundError(path)

    return (gt > 0).astype(np.uint8)


def softmax_shadow_prob(seg_logits_2hw, alpha):

    logits = seg_logits_2hw.float() * float(alpha)

    probs = F.softmax(logits, dim=0)

    return probs[1].detach().cpu().numpy().astype(np.float32)


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


def apply_postproc(mask01, min_area, close_k):

    m = connected_components_filter(mask01, min_area)

    m = morph_close(m, close_k)

    return m


def metrics(pred01, gt01):

    pred = pred01.astype(bool)

    gt = gt01.astype(bool)

    tp = np.logical_and(pred, gt).sum()

    fp = np.logical_and(pred, ~gt).sum()

    fn = np.logical_and(~pred, gt).sum()

    iou = tp / (tp + fp + fn + 1e-9)

    f1 = (2 * tp) / (2 * tp + fp + fn + 1e-9)

    return int(tp), int(fp), int(fn), float(iou), float(f1)


def overlay_red(image_bgr, mask01, alpha=0.45):

    out = image_bgr.copy()

    red = np.zeros_like(out)

    red[:, :, 2] = 255

    m = mask01.astype(bool)

    out[m] = (out[m] * (1 - alpha) + red[m] * alpha).astype(np.uint8)

    return out


def save_prob_npz(prob, out_path):

    np.savez_compressed(out_path, prob=prob)


def load_prob_npz(path):

    return np.load(path)["prob"]


def thresholds(tmin, tmax, tstep):

    vals = []

    t = tmin

    while t <= tmax + 1e-9:

        vals.append(round(float(t), 6))

        t += tstep

    return vals


def cache_probs(model, img_paths, alpha, probs_dir):

    ensure_dir(probs_dir)

    cached = []

    for i, ip in enumerate(img_paths):

        stem = os.path.splitext(os.path.basename(ip))[0]

        outp = os.path.join(probs_dir, f"{stem}.npz")

        if os.path.exists(outp):

            cached.append(outp)

            continue

        print(
            f"[cache_probs] alpha={alpha} {i+1}/{len(img_paths)} {os.path.basename(ip)}",
            flush=True,
        )

        res = inference_model(model, ip)

        prob = softmax_shadow_prob(res.seg_logits.data, alpha=alpha)

        save_prob_npz(prob, outp)

        cached.append(outp)

    return cached


def eval_from_probs(prob_paths, gt_paths, thr, min_area, close_k):

    TP = FP = FN = 0

    for pp, gp in zip(prob_paths, gt_paths):

        prob = load_prob_npz(pp)

        pred01 = (prob >= thr).astype(np.uint8)

        pred01 = apply_postproc(pred01, min_area=min_area, close_k=close_k)

        gt01 = load_gt_mask01(gp)

        tp, fp, fn, _, _ = metrics(pred01, gt01)

        TP += tp

        FP += fp

        FN += fn

    iou = TP / (TP + FP + FN + 1e-9)

    f1 = (2 * TP) / (2 * TP + FP + FN + 1e-9)

    return {"TP": TP, "FP": FP, "FN": FN, "IoU": float(iou), "F1": float(f1)}


def run_domain(
    name,
    model,
    img_dir,
    gt_dir,
    img_ext,
    out_root,
    limit,
    alpha_list,
    thr_list,
    min_areas,
    close_ks,
):

    imgs = list_images(img_dir, img_ext, limit)

    pairs = []

    for ip in imgs:

        gp = gt_for_image(gt_dir, ip)

        if gp:

            pairs.append((ip, gp))

    if not pairs:

        raise RuntimeError(f"[{name}] no image/gt pairs found.")

    img_paths = [p[0] for p in pairs]

    gt_paths = [p[1] for p in pairs]

    domain_out = os.path.join(out_root, name)

    probs_base = os.path.join(domain_out, "probs")

    ensure_dir(domain_out)

    ensure_dir(probs_base)

    all_records = []

    best = None

    for alpha in alpha_list:

        probs_dir = os.path.join(probs_base, f"alpha_{str(alpha).replace('.', 'p')}")

        prob_paths = cache_probs(model, img_paths, alpha=alpha, probs_dir=probs_dir)

        for thr, min_area, close_k in itertools.product(thr_list, min_areas, close_ks):

            m = eval_from_probs(prob_paths, gt_paths, thr, min_area, close_k)

            rec = {
                "domain": name,
                "alpha": float(alpha),
                "t": float(thr),
                "min_area": int(min_area),
                "close_k": int(close_k),
                **m,
            }

            all_records.append(rec)

            if best is None or rec["IoU"] > best["IoU"]:

                best = rec

        print(f"[{name}] finished alpha={alpha}")

    return best, all_records


def parse_floats(s):

    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_ints(s):

    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--device", default="cpu")

    ap.add_argument("--cfg", required=True)

    ap.add_argument("--ckpt", required=True)

    ap.add_argument("--out-root", default=r"C:\RoverShadowFinal\out")

    ap.add_argument("--priv-img-dir", required=True)

    ap.add_argument("--priv-gt-dir", required=True)

    ap.add_argument("--priv-ext", default=".png")

    ap.add_argument("--priv-limit", type=int, default=0)

    ap.add_argument("--pub-img-dir", required=True)

    ap.add_argument("--pub-gt-dir", required=True)

    ap.add_argument("--pub-ext", default=".jpg")

    ap.add_argument("--pub-limit", type=int, default=0)

    ap.add_argument("--alphas", default="0.5,1.0,1.5,2.0")

    ap.add_argument("--min-areas", default="0,100,200,500")

    ap.add_argument("--close-ks", default="0,3,5")

    ap.add_argument("--priv-tmin", type=float, default=0.70)

    ap.add_argument("--priv-tmax", type=float, default=0.95)

    ap.add_argument("--priv-tstep", type=float, default=0.05)

    ap.add_argument("--pub-tmin", type=float, default=0.30)

    ap.add_argument("--pub-tmax", type=float, default=0.70)

    ap.add_argument("--pub-tstep", type=float, default=0.05)

    args = ap.parse_args()

    ensure_dir(args.out_root)

    ensure_dir(os.path.join(args.out_root, "reports"))

    cfg = Config.fromfile(args.cfg)

    model = init_model(cfg, args.ckpt, device=args.device)

    alpha_list = parse_floats(args.alphas)

    min_areas = parse_ints(args.min_areas)

    close_ks = parse_ints(args.close_ks)

    priv_thr = thresholds(args.priv_tmin, args.priv_tmax, args.priv_tstep)

    pub_thr = thresholds(args.pub_tmin, args.pub_tmax, args.pub_tstep)

    t0 = time.time()

    best_priv, rec_priv = run_domain(
        "private",
        model,
        args.priv_img_dir,
        args.priv_gt_dir,
        args.priv_ext,
        args.out_root,
        args.priv_limit,
        alpha_list,
        priv_thr,
        min_areas,
        close_ks,
    )

    best_pub, rec_pub = run_domain(
        "public_val",
        model,
        args.pub_img_dir,
        args.pub_gt_dir,
        args.pub_ext,
        args.out_root,
        args.pub_limit,
        alpha_list,
        pub_thr,
        min_areas,
        close_ks,
    )

    grid = rec_priv + rec_pub

    with open(os.path.join(args.out_root, "reports", "grid_search.json"), "w") as f:

        json.dump(grid, f, indent=2)

    best_params = {"private": best_priv, "public_val": best_pub}

    with open(os.path.join(args.out_root, "reports", "best_params.json"), "w") as f:

        json.dump(best_params, f, indent=2)

    print("\n=== BEST PRIVATE ===")

    print(best_priv)

    print("\n=== BEST PUBLIC_VAL ===")

    print(best_pub)

    print("\n[INFO] seconds:", round(time.time() - t0, 2))


if __name__ == "__main__":

    main()
