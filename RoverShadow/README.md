# RoverShadow

Binary shadow segmentation project built on MMSegmentation, with DeepLabV3+ and FCN experiment configs, calibration scripts, and a local ShadowSeg demo UI.

## Environment

Use one requirements file:

- CPU: `requirements-cpu.txt`
- GPU (CUDA 13.0): `requirements-gpu-cu130.txt`

Example:

```powershell
python -m pip install -r requirements-cpu.txt
```

## Locked Demo Model

- Config: `configs/shadow_deeplabv3plus_r50_run_c_aug_fp.py`
- Checkpoint: `work_dirs/phase2_r50_best_long12000/iter_11000.pth`

## Data Paths

Expected dataset structure:

- `data/public/Rover_Shadow_Public_Dataset/ShadowImages/train`
- `data/public/Rover_Shadow_Public_Dataset/ShadowMasks/train`
- `data/public/Rover_Shadow_Public_Dataset/ShadowImages/val`
- `data/public/Rover_Shadow_Public_Dataset/ShadowMasks/val`
- `data/private/LunarShadowDataset/ShadowImages`
- `data/private/LunarShadowDataset/ShadowMasks`

## Key Scripts

- `scripts/infer_deeplab_one.py`: single-image inference (mask + overlay)
- `scripts/infer_cache_folder.py`: cache probability maps for image folders
- `scripts/render_cache_folder.py`: render masks/overlays from cached probabilities
- `scripts/eval_iou_folder.py`: folder-level IoU evaluation
- `scripts/sweep_alpha_threshold_iou.py`: alpha/threshold sweep
- `scripts/pipeline_all.py`: end-to-end pipeline runner

## Quick CLI Examples

Single image inference:

```powershell
python scripts/infer_deeplab_one.py --cfg configs/shadow_deeplabv3plus_r50_run_c_aug_fp.py --ckpt work_dirs/phase2_r50_best_long12000/iter_11000.pth --img data/public/Rover_Shadow_Public_Dataset/ShadowImages/val/lssd4000.jpg --out outputs/demo/overlays/lssd4000_overlay.png --out-mask outputs/demo/masks/lssd4000_mask.png --device cpu --threshold 0.25 --min-area 0 --close-k 7 --save-prob
```

Sweep alpha/threshold:

```powershell
python scripts/sweep_alpha_threshold_iou.py --cfg configs/shadow_deeplabv3plus_r50_run_c_aug_fp.py --ckpt work_dirs/phase2_r50_best_long12000/iter_11000.pth --img-dir data/public/Rover_Shadow_Public_Dataset/ShadowImages/val --gt-dir data/public/Rover_Shadow_Public_Dataset/ShadowMasks/val --ext .png --tmin 0.2 --tmax 0.95 --tstep 0.05 --alphas 1.0 --min-area 0 --close-k 7 --device cpu
```

## ShadowSeg UI

Backend:

```powershell
cd C:\RoverShadowFinal\RoverShadow
python frontend\backend.py
```

Frontend:

```powershell
cd C:\RoverShadowFinal\RoverShadow\frontend
npm install
npm run dev
```

Open:

- `http://127.0.0.1:5173`
- `http://127.0.0.1:8000/health`

## Notes

- `outputs/demo/overlays` and `outputs/demo/masks` are used for demo outputs.
- `mmcv-lite` compatibility is handled in `rovershadow/runtime/mmcv_ops_shim.py`.
