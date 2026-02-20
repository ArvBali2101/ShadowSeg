from __future__ import annotations

import argparse

from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:

    sys.path.insert(0, str(ROOT))

import rovershadow.losses

from mmengine.config import Config

from mmseg.apis import init_model, inference_model

from mmseg.apis.inference import show_result_pyplot


def main() -> None:

    p = argparse.ArgumentParser(description="CPU smoke inference run")

    p.add_argument(
        "--cfg",
        default=str(
            ROOT / "models/deeplab/configs/shadow_deeplabv3plus_r50_run_c_aug_fp.py"
        ),
    )

    p.add_argument(
        "--ckpt",
        default=str(
            ROOT / "models/deeplab/work_dirs/phase2_r50_best_long12000/iter_11000.pth"
        ),
    )

    p.add_argument(
        "--img",
        default=str(ROOT / "data/private/LunarShadowDataset/ShadowImages/Image-1.png"),
    )

    p.add_argument("--out", default=str(ROOT / "outputs/smoke_cpu_overlay.png"))

    a = p.parse_args()

    cfg = Config.fromfile(a.cfg)

    model = init_model(cfg, a.ckpt, device="cpu")

    result = inference_model(model, a.img)

    out = Path(a.out)

    out.parent.mkdir(parents=True, exist_ok=True)

    show_result_pyplot(
        model, a.img, result, show=False, out_file=str(out), opacity=0.55
    )

    print(f"[OK] wrote {out}")


if __name__ == "__main__":

    main()
