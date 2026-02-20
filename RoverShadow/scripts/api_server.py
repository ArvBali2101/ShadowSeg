import os, uuid, json

import numpy as np

import cv2

import torch

import torch.nn.functional as F

from fastapi import FastAPI

from pydantic import BaseModel

from mmengine.config import Config

from mmseg.apis import init_model, inference_model

APP_ROOT = r"C:\RoverShadowFinal"

OUT_ROOT = os.path.join(APP_ROOT, "out")

REPORTS = os.path.join(OUT_ROOT, "reports")

CACHE = os.path.join(OUT_ROOT, "api_cache")

os.makedirs(CACHE, exist_ok=True)

CFG_PATH = (
    r"C:\RoverShadowFinal\RoverShadow\configs\shadow_deeplabv3plus_r50_run_c_aug_fp.py"
)

CKPT_PATH = r"C:\RoverShadowFinal\RoverShadow\work_dirs\phase2_r50_best_long12000\iter_11000.pth"

DEVICE = "cpu"

app = FastAPI(title="RoverShadow API")

cfg = Config.fromfile(CFG_PATH)

model = init_model(cfg, CKPT_PATH, device=DEVICE)


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


def overlay_red(image_bgr, mask01, alpha=0.45):

    out = image_bgr.copy()

    red = np.zeros_like(out)
    red[:, :, 2] = 255

    m = mask01.astype(bool)

    out[m] = (out[m] * (1 - alpha) + red[m] * alpha).astype(np.uint8)

    return out


class InferOnceReq(BaseModel):

    img_path: str

    alpha: float = 1.0


class InferOnceResp(BaseModel):

    id: str

    prob_path: str

    img_path: str


class RenderReq(BaseModel):

    id: str

    t: float

    min_area: int = 0

    close_k: int = 0


class RenderResp(BaseModel):

    mask_path: str

    overlay_path: str


@app.get("/best_params")
def best_params():

    p = os.path.join(REPORTS, "best_params.json")

    if not os.path.exists(p):

        return {"error": "best_params.json not found. Run pipeline_all.py once."}

    return json.load(open(p, "r", encoding="utf-8"))


@app.post("/infer_once", response_model=InferOnceResp)
def infer_once(req: InferOnceReq):

    if not os.path.exists(req.img_path):

        return {"id": "", "prob_path": "", "img_path": req.img_path}

    rid = str(uuid.uuid4())[:8]

    prob_path = os.path.join(CACHE, f"{rid}.npz")

    res = inference_model(model, req.img_path)

    prob = softmax_shadow_prob(res.seg_logits.data, req.alpha)

    np.savez_compressed(prob_path, prob=prob)

    return {"id": rid, "prob_path": prob_path, "img_path": req.img_path}


@app.post("/render", response_model=RenderResp)
def render(req: RenderReq):

    prob_path = os.path.join(CACHE, f"{req.id}.npz")

    if not os.path.exists(prob_path):

        return {"mask_path": "", "overlay_path": ""}

    prob = np.load(prob_path)["prob"]

    mask01 = (prob >= req.t).astype(np.uint8)

    mask01 = connected_components_filter(mask01, req.min_area)

    mask01 = morph_close(mask01, req.close_k)

    mask_path = os.path.join(CACHE, f"{req.id}_mask.png")

    cv2.imwrite(mask_path, (mask01 * 255).astype(np.uint8))

    meta_path = os.path.join(CACHE, f"{req.id}.json")

    if os.path.exists(meta_path):

        img_path = json.load(open(meta_path, "r", encoding="utf-8"))["img_path"]

    else:

        img_path = None

    if img_path and os.path.exists(img_path):

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        ov = overlay_red(img, mask01)

        overlay_path = os.path.join(CACHE, f"{req.id}_overlay.png")

        cv2.imwrite(overlay_path, ov)

    else:

        overlay_path = ""

    return {"mask_path": mask_path, "overlay_path": overlay_path}
