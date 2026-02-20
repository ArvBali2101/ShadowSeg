from __future__ import annotations

from pathlib import Path

import os

import sys

from mmengine.config import Config

from mmseg.apis import init_model, inference_model

from mmseg.apis.inference import show_result_pyplot

ROOT = Path(__file__).resolve().parents[1]


def die(msg: str, code: int = 1) -> None:

    print(f"[ERROR] {msg}")

    raise SystemExit(code)


def main() -> None:

    img_path = ROOT / "data/private/LunarShadowDataset/ShadowImages/Image-2.png"

    cfg_path = ROOT / "models/fcn/configs/shadow_fcn_r50.py"

    ckpt_path = ROOT / "models/fcn/work_dirs/shadow_fcn_r50_gpu/iter_5000.pth"

    out_path = ROOT / "outputs/fcn_overlay.png"

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

            die("Could not find test pipeline in config.")

    model = init_model(cfg, str(ckpt_path), device="cpu")

    result = inference_model(model, str(img_path))

    show_result_pyplot(
        model, str(img_path), result, show=False, out_file=str(out_path), opacity=0.6
    )

    print("[OK] Saved:", out_path)


if __name__ == "__main__":

    main()
