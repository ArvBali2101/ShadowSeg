ShadowSeg (RoverShadow)

ShadowSeg is a binary semantic segmentation system for shadow detection, built using MMSegmentation with DeepLabV3+ and FCN models.

The project focuses on:

understanding shadow perception under challenging lighting conditions
building a full ML pipeline for training, evaluation, and inference
improving segmentation performance through calibration and post-processing
A local demo UI is included for interactive testing.

Overview
This project implements an end-to-end workflow for shadow segmentation:

model training and evaluation
probability calibration and threshold tuning
post-processing for cleaner segmentation outputs
visualization through overlays and UI tools
The system is designed to explore how segmentation models behave under different lighting conditions and domain shifts.

Environment
Use one requirements file:

CPU: requirements-cpu.txt
GPU (CUDA 13.0): requirements-gpu-cu130.txt
Example:

python -m pip install -r requirements-cpu.txt
Locked Demo Model
Config:
configs/shadow_deeplabv3plus_r50_run_c_aug_fp.py
Checkpoint:
work_dirs/phase2_r50_best_long12000/iter_11000.pth
Data Paths

Expected dataset structure:

data/public/Rover_Shadow_Public_Dataset/ShadowImages/train
data/public/Rover_Shadow_Public_Dataset/ShadowMasks/train
data/public/Rover_Shadow_Public_Dataset/ShadowImages/val
data/public/Rover_Shadow_Public_Dataset/ShadowMasks/val

data/private/LunarShadowDataset/ShadowImages
data/private/LunarShadowDataset/ShadowMasks
Key Scripts
scripts/infer_deeplab_one.py → single-image inference (mask + overlay)
scripts/infer_cache_folder.py → cache probability maps for folders
scripts/render_cache_folder.py → render masks/overlays from cached probabilities
scripts/eval_iou_folder.py → folder-level IoU evaluation
scripts/sweep_alpha_threshold_iou.py → threshold / calibration sweep
scripts/pipeline_all.py → end-to-end pipeline runner
Quick CLI Examples
Single Image Inference
python scripts/infer_deeplab_one.py \
  --cfg configs/shadow_deeplabv3plus_r50_run_c_aug_fp.py \
  --ckpt work_dirs/phase2_r50_best_long12000/iter_11000.pth \
  --img data/public/Rover_Shadow_Public_Dataset/ShadowImages/val/lssd4000.jpg \
  --out outputs/demo/overlays/lssd4000_overlay.png \
  --out-mask outputs/demo/masks/lssd4000_mask.png \
  --device cpu \
  --threshold 0.25 \
  --min-area 0 \
  --close-k 7 \
  --save-prob
Threshold / Calibration Sweep
python scripts/sweep_alpha_threshold_iou.py \
  --cfg configs/shadow_deeplabv3plus_r50_run_c_aug_fp.py \
  --ckpt work_dirs/phase2_r50_best_long12000/iter_11000.pth \
  --img-dir data/public/Rover_Shadow_Public_Dataset/ShadowImages/val \
  --gt-dir data/public/Rover_Shadow_Public_Dataset/ShadowMasks/val \
  --ext .png \
  --tmin 0.2 \
  --tmax 0.95 \
  --tstep 0.05 \
  --alphas 1.0 \
  --min-area 0 \
  --close-k 7 \
  --device cpu
ShadowSeg UI
Backend
conda activate shadowseg
cd RoverShadow
python frontend/backend.py
Frontend
conda activate shadowseg
cd RoverShadow/frontend
npm install
npm run dev
Open in Browser
Frontend: http://127.0.0.1:5173
Backend health: http://127.0.0.1:8000/health
Notes
outputs/demo/overlays → visualization outputs
outputs/demo/masks → predicted masks
mmcv-lite compatibility handled in:
rovershadow/runtime/mmcv_ops_shim.py
