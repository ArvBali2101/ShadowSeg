import io

import json

import sys

import uuid

from pathlib import Path

import numpy as np

from PIL import Image

from fastapi import FastAPI, UploadFile, File, Form

from fastapi.responses import JSONResponse

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:

    sys.path.insert(0, str(REPO_ROOT))

import rovershadow.losses

from rovershadow.utils.render import prob_to_mask01

from mmengine.config import Config

from mmseg.apis import init_model, inference_model

APP = FastAPI(title="RoverShadow API", version="1.0")

CACHE_DIR = REPO_ROOT / "cache_api"

CACHE_DIR.mkdir(parents=True, exist_ok=True)

BEST_PARAMS_PATH = REPO_ROOT / "api" / "best_params.json"

MODEL = None

MODEL_CFG = None

MODEL_CKPT = None


def softmax_prob_class1(seg_logits: np.ndarray) -> np.ndarray:

    seg_logits = seg_logits.astype(np.float32)

    seg_logits = seg_logits - seg_logits.max(axis=0, keepdims=True)

    exp = np.exp(seg_logits)

    prob1 = exp[1] / (exp.sum(axis=0) + 1e-9)

    return prob1.astype(np.float32)


def ensure_model(cfg_path: Path, ckpt_path: Path, device: str = "cpu"):

    global MODEL, MODEL_CFG, MODEL_CKPT

    cfg_path = cfg_path.resolve()

    ckpt_path = ckpt_path.resolve()

    if MODEL is not None and MODEL_CFG == cfg_path and MODEL_CKPT == ckpt_path:

        return MODEL

    cfg = Config.fromfile(str(cfg_path))

    MODEL = init_model(cfg, str(ckpt_path), device=device)

    MODEL_CFG = cfg_path

    MODEL_CKPT = ckpt_path

    return MODEL


@APP.get("/best_params")
def best_params():

    if BEST_PARAMS_PATH.exists():

        return json.loads(BEST_PARAMS_PATH.read_text(encoding="utf-8"))

    return {
        "public": {"threshold": 0.25, "close_k": 7, "min_area": 0},
        "private": {"threshold": 0.90, "close_k": 7, "min_area": 0},
    }


@APP.post("/infer_once")
async def infer_once(
    file: UploadFile = File(...),
    cfg: str = Form(...),
    ckpt: str = Form(...),
    device: str = Form("cpu"),
):

    img_bytes = await file.read()

    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    job_id = str(uuid.uuid4())

    job_dir = CACHE_DIR / job_id

    job_dir.mkdir(parents=True, exist_ok=True)

    img_path = job_dir / "image.png"

    img.save(img_path)

    model = ensure_model(REPO_ROOT / cfg, REPO_ROOT / ckpt, device=device)

    result = inference_model(model, str(img_path))

    if not hasattr(result, "seg_logits") or result.seg_logits is None:

        return JSONResponse(
            {"error": "seg_logits not found in inference result"}, status_code=500
        )

    seg_logits = result.seg_logits.data.detach().cpu().numpy()

    prob = softmax_prob_class1(seg_logits)

    np.savez_compressed(job_dir / "prob.npz", prob=prob)

    return {"job_id": job_id}


@APP.post("/render")
async def render(
    job_id: str = Form(...),
    threshold: float = Form(...),
    min_area: int = Form(0),
    close_k: int = Form(0),
    alpha: float = Form(0.55),
):

    job_dir = CACHE_DIR / job_id

    prob_npz = job_dir / "prob.npz"

    img_path = job_dir / "image.png"

    if not prob_npz.exists() or not img_path.exists():

        return JSONResponse({"error": "job_id not found"}, status_code=404)

    prob = np.load(prob_npz)["prob"].astype(np.float32)

    mask01 = prob_to_mask01(
        prob, threshold=threshold, min_area=min_area, close_k=close_k
    )

    img = Image.open(img_path).convert("RGB")

    img_rgb = np.array(img).astype(np.float32)

    m = (mask01 > 0).astype(np.float32)[..., None]

    red = np.zeros_like(img_rgb)

    red[..., 0] = 255.0

    out = img_rgb * (1.0 - alpha * m) + red * (alpha * m)

    out = np.clip(out, 0, 255).astype(np.uint8)

    out_mask = job_dir / "mask.png"

    out_overlay = job_dir / "overlay.png"

    Image.fromarray((mask01 * 255).astype(np.uint8)).save(out_mask)

    Image.fromarray(out).save(out_overlay)

    return {"mask_path": str(out_mask), "overlay_path": str(out_overlay)}
