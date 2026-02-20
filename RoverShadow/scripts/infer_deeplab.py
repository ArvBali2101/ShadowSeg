from __future__ import annotations

import argparse

from pathlib import Path

import os

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(0, str(PROJECT_ROOT))

from rovershadow.runtime import ensure_runtime_env, install_mmcv_ops_shim_if_needed

ensure_runtime_env()

install_mmcv_ops_shim_if_needed()

import rovershadow.losses

import torch

from mmengine.config import Config

from mmseg.apis import inference_model, init_model

from mmseg.apis.inference import show_result_pyplot


def die(msg: str, code: int = 1) -> None:

    print(f"[ERROR] {msg}")

    raise SystemExit(code)


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description="Run DeepLab inference (CPU/GPU).")

    parser.add_argument(
        "--img",
        default=str(
            PROJECT_ROOT / "data/private/LunarShadowDataset/ShadowImages/Image-1.png"
        ),
    )

    parser.add_argument(
        "--cfg",
        default=str(
            PROJECT_ROOT
            / "models/deeplab/configs/shadow_deeplabv3plus_r50_run_c_aug_fp.py"
        ),
    )

    parser.add_argument(
        "--ckpt",
        default=str(
            PROJECT_ROOT
            / "models/deeplab/work_dirs/phase2_r50_best_long12000/iter_11000.pth"
        ),
    )

    parser.add_argument(
        "--out", default=str(PROJECT_ROOT / "outputs/deeplab_overlay.png")
    )

    parser.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda"])

    return parser.parse_args()


def resolve_device(requested: str) -> str:

    if requested == "cpu":

        return "cpu"

    if requested == "cuda":

        if not torch.cuda.is_available():

            die("CUDA requested but not available.")

        return "cuda"

    return "cuda" if torch.cuda.is_available() else "cpu"


def main() -> None:

    args = parse_args()

    img_path = Path(args.img)

    cfg_path = Path(args.cfg)

    ckpt_path = Path(args.ckpt)

    out_path = Path(args.out)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("[INFO] Current working directory:", os.getcwd())

    print("[INFO] img:", img_path)

    print("[INFO] cfg:", cfg_path)

    print("[INFO] ckpt:", ckpt_path)

    print("[INFO] out:", out_path)

    if not img_path.is_file():

        die(f"Image not found: {img_path}")

    if not cfg_path.is_file():

        die(f"Config not found: {cfg_path}")

    if not ckpt_path.is_file():

        die(f"Checkpoint not found: {ckpt_path}")

    cfg = Config.fromfile(str(cfg_path))

    if not hasattr(cfg, "test_pipeline"):

        if (
            "test_dataloader" in cfg
            and "dataset" in cfg.test_dataloader
            and "pipeline" in cfg.test_dataloader.dataset
        ):

            cfg.test_pipeline = cfg.test_dataloader.dataset.pipeline

        elif (
            "val_dataloader" in cfg
            and "dataset" in cfg.val_dataloader
            and "pipeline" in cfg.val_dataloader.dataset
        ):

            cfg.test_pipeline = cfg.val_dataloader.dataset.pipeline

        else:

            die("Could not find test pipeline.")

    model = init_model(cfg, str(ckpt_path), device=resolve_device(args.device))

    result = inference_model(model, str(img_path))

    show_result_pyplot(
        model, str(img_path), result, show=False, out_file=str(out_path), opacity=0.6
    )

    print("[OK] Saved:", out_path)


if __name__ == "__main__":

    main()
