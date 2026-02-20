from __future__ import annotations

import hashlib

import os

import re

import subprocess

import sys

from datetime import datetime

from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile

from fastapi.middleware.cors import CORSMiddleware

from fastapi.responses import JSONResponse

from fastapi.staticfiles import StaticFiles

root = Path(__file__).resolve().parents[1]

script_path = root / "scripts" / "infer_deeplab_one.py"

config_path = root / "configs" / "shadow_deeplabv3plus_r50_run_c_aug_fp.py"

ckpt_path = root / "work_dirs" / "phase2_r50_best_long12000" / "iter_11000.pth"

outputs_root = root / "outputs" / "demo"

upload_dir = outputs_root / "uploads"

overlays_dir = outputs_root / "overlays"

masks_dir = outputs_root / "masks"

prob_dir = outputs_root / "prob"

for p in (upload_dir, overlays_dir, masks_dir, prob_dir):

    p.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="ShadowSeg Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/outputs", StaticFiles(directory=str(outputs_root)), name="outputs")


def sanitize_stem(name: str) -> str:

    stem = Path(name).stem

    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", stem)

    cleaned = re.sub(r"_+", "_", cleaned).strip("._")

    return cleaned[:120] or "upload"


def ext_from_name(name: str) -> str:

    suffix = Path(name).suffix.lower()

    if suffix in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:

        return suffix

    return ".png"


def build_run_id(
    name: str, payload: bytes, threshold: float, close_k: int, min_area: int
) -> str:

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    h = hashlib.sha1()

    h.update(name.encode("utf-8", errors="ignore"))

    h.update(payload)

    h.update(f"{threshold:.2f}|{close_k}|{min_area}".encode("utf-8"))

    return f"{stamp}_{h.hexdigest()[:10]}"


def image_files(path: Path) -> list[Path]:

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

    return sorted(
        [p for p in path.glob("*") if p.is_file() and p.suffix.lower() in exts]
    )


def url_for_demo_path(path: Path) -> str:

    rel = path.relative_to(outputs_root).as_posix()

    return f"/outputs/{rel}"


@app.get("/health")
def health():

    env = os.environ.get("CONDA_DEFAULT_ENV", "")

    state = {
        "status": "ok",
        "env": env,
        "script_exists": script_path.exists(),
        "config_exists": config_path.exists(),
        "ckpt_exists": ckpt_path.exists(),
    }

    if (
        not state["script_exists"]
        or not state["config_exists"]
        or not state["ckpt_exists"]
    ):

        return JSONResponse(status_code=500, content={"status": "error", **state})

    return state


@app.get("/api/default-example")
def default_example():

    uploads = image_files(upload_dir)

    for img in uploads:

        stem = img.stem

        overlay = overlays_dir / f"{stem}_overlay.png"

        mask = masks_dir / f"{stem}_mask.png"

        if overlay.exists() and mask.exists():

            return {
                "status": "ok",
                "filename": img.name,
                "original_url": url_for_demo_path(img),
                "overlay_url": url_for_demo_path(overlay),
                "mask_url": url_for_demo_path(mask),
            }

    overlays = sorted(overlays_dir.glob("*.png"))

    masks = sorted(masks_dir.glob("*.png"))

    if overlays:

        overlay = overlays[0]

        stem = overlay.stem.removesuffix("_overlay")

        same_mask = masks_dir / f"{stem}_mask.png"

        mask = same_mask if same_mask.exists() else (masks[0] if masks else overlay)

        return {
            "status": "ok",
            "filename": overlay.name,
            "original_url": url_for_demo_path(overlay),
            "overlay_url": url_for_demo_path(overlay),
            "mask_url": url_for_demo_path(mask),
        }

    return {"status": "empty"}


@app.post("/api/run")
async def run_shadowseg(
    file: UploadFile = File(...),
    threshold: float = Form(...),
    min_area: int = Form(0),
    close_k: int = Form(7),
):

    name = file.filename or "upload.png"

    payload = await file.read()

    if not payload:

        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": "Uploaded file is empty."},
        )

    if threshold < 0.0 or threshold > 1.0:

        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": "threshold must be in [0.0, 1.0]."},
        )

    run_id = build_run_id(name, payload, float(threshold), int(close_k), int(min_area))

    stem = sanitize_stem(Path(name).stem)

    base_name = f"{stem}_{run_id}"

    suffix = ext_from_name(name)

    upload_path = upload_dir / f"{base_name}{suffix}"

    overlay_path = overlays_dir / f"{base_name}_overlay.png"

    mask_path = masks_dir / f"{base_name}_mask.png"

    prob_path = prob_dir / f"{base_name}_prob.npz"

    upload_path.write_bytes(payload)

    cmd = [
        sys.executable,
        str(script_path),
        "--cfg",
        str(config_path),
        "--ckpt",
        str(ckpt_path),
        "--img",
        str(upload_path),
        "--out",
        str(overlay_path),
        "--out-mask",
        str(mask_path),
        "--out-prob",
        str(prob_path),
        "--device",
        "cpu",
        "--threshold",
        f"{float(threshold):.2f}",
        "--min-area",
        str(int(min_area)),
        "--close-k",
        str(int(close_k)),
        "--save-prob",
    ]

    proc = subprocess.run(
        cmd, cwd=str(root), capture_output=True, text=True, check=False
    )

    if proc.returncode != 0:

        detail = (proc.stderr or proc.stdout or "").strip()

        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": "Inference failed.",
                "detail": detail,
                "code": proc.returncode,
            },
        )

    if not overlay_path.exists() or not mask_path.exists():

        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": "Inference completed but output files were not created.",
            },
        )

    return {
        "status": "ok",
        "run_id": run_id,
        "filename": name,
        "params": {
            "threshold": float(threshold),
            "close_k": int(close_k),
            "min_area": int(min_area),
        },
        "original_url": url_for_demo_path(upload_path),
        "overlay_url": url_for_demo_path(overlay_path),
        "mask_url": url_for_demo_path(mask_path),
        "prob_url": url_for_demo_path(prob_path),
    }


if __name__ == "__main__":

    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)
