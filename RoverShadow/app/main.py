from pathlib import Path

import re

import subprocess

import sys

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

for p in [upload_dir, overlays_dir, masks_dir, prob_dir]:

    p.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="ShadowSeg Local Runner", version="1.0.0")

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

    if not cleaned:

        return "upload"

    return cleaned[:120]


def ext_from_name(name: str) -> str:

    suffix = Path(name).suffix.lower()

    if suffix in {".jpg", ".jpeg", ".png"}:

        return suffix

    return ".png"


@app.get("/health")
def health():

    data = {
        "status": "ok",
        "python": sys.executable,
        "conda_env": str(__import__("os").environ.get("CONDA_DEFAULT_ENV", "")),
        "script_exists": script_path.exists(),
        "config_exists": config_path.exists(),
        "ckpt_exists": ckpt_path.exists(),
    }

    if not data["script_exists"]:

        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Missing script: {script_path}",
                **data,
            },
        )

    if not data["config_exists"]:

        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Missing config: {config_path}",
                **data,
            },
        )

    if not data["ckpt_exists"]:

        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Missing checkpoint: {ckpt_path}",
                **data,
            },
        )

    return data


@app.post("/api/run")
async def run_shadowseg(
    file: UploadFile = File(...),
    threshold: float = Form(...),
    min_area: int = Form(0),
    close_k: int = Form(7),
):

    name = file.filename or "upload.png"

    suffix = ext_from_name(name)

    if suffix not in {".jpg", ".jpeg", ".png"}:

        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "message": "Only jpg, jpeg, png files are supported.",
            },
        )

    stem = sanitize_stem(name)

    upload_path = upload_dir / f"{stem}{suffix}"

    overlay_path = overlays_dir / f"{stem}_overlay.png"

    mask_path = masks_dir / f"{stem}_mask.png"

    prob_path = prob_dir / f"{stem}_prob.npz"

    content = await file.read()

    if not content:

        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": "Uploaded file is empty."},
        )

    upload_path.write_bytes(content)

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
        str(threshold),
        "--min-area",
        str(min_area),
        "--close-k",
        str(close_k),
        "--save-prob",
    ]

    try:

        proc = subprocess.run(
            cmd,
            cwd=str(root),
            capture_output=True,
            text=True,
            check=False,
        )

    except FileNotFoundError:

        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": "Python runtime not found. Activate your conda environment and retry.",
            },
        )

    except Exception as e:

        return JSONResponse(
            status_code=500, content={"status": "error", "message": str(e)}
        )

    if proc.returncode != 0:

        stderr = (proc.stderr or "").strip()

        stdout = (proc.stdout or "").strip()

        detail = stderr if stderr else stdout

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
        "file_stem": stem,
        "params": {"threshold": threshold, "close_k": close_k, "min_area": min_area},
        "original_url": f"/outputs/uploads/{upload_path.name}",
        "overlay_url": f"/outputs/overlays/{overlay_path.name}",
        "mask_url": f"/outputs/masks/{mask_path.name}",
        "prob_url": f"/outputs/prob/{prob_path.name}",
        "stdout": (proc.stdout or "").strip(),
    }


if __name__ == "__main__":

    try:

        import uvicorn

    except Exception:

        raise SystemExit("uvicorn is missing. Install backend requirements first.")

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False)
