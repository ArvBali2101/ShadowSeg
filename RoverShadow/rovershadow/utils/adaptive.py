from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AdaptiveV2Config:
    """
    Adaptive thresholding based on calibration statistics of the probability map.

    We avoid area-prior quantiles (too brittle across domains).
    Instead we use simple, interpretable statistics:

      mean_p: average shadow probability
      p95: 95th percentile (how high the confident tail is)

    Policy idea:
      - If mean is tiny and tail isn't strong -> model is uncertain -> use high threshold (reduce FP)
      - If tail is strong -> allow lower threshold (keep recall)
    """

    thr_min: float = 0.20

    thr_max: float = 0.99


def clamp(x: float, lo: float, hi: float) -> float:

    return float(max(lo, min(hi, x)))


def stats(prob: np.ndarray) -> tuple[float, float]:

    p = np.asarray(prob, dtype=np.float32)

    if p.ndim != 2:

        raise ValueError(f"prob must be HxW, got shape={p.shape}")

    mean_p = float(np.mean(p))

    p95 = float(np.quantile(p, 0.95))

    return mean_p, p95


def choose_threshold_v2(prob: np.ndarray, domain: str = "auto") -> float:
    """
    Returns an adaptive threshold based on prob distribution.

    Defaults are chosen to preserve your strong PUBLIC setting (~0.25)
    while being more conservative on PRIVATE to reduce false positives.
    """

    mean_p, p95 = stats(prob)

    d = (domain or "auto").lower()

    if d == "public":

        if mean_p < 0.04 and p95 < 0.55:

            thr = 0.35

        elif mean_p < 0.03 and p95 < 0.45:

            thr = 0.45

        else:

            thr = 0.25

        return clamp(thr, 0.20, 0.60)

    if d == "private":

        if mean_p < 0.03:

            thr = 0.90

        elif mean_p < 0.05:

            thr = 0.80

        elif mean_p < 0.08:

            thr = 0.70

        else:

            thr = 0.55

        if p95 > 0.92:

            thr = min(thr, 0.70)

        if p95 > 0.97:

            thr = min(thr, 0.60)

        return clamp(thr, 0.40, 0.99)

    if mean_p < 0.03:

        thr = 0.75

    elif mean_p < 0.06:

        thr = 0.60

    else:

        thr = 0.45

    return clamp(thr, 0.25, 0.95)


def choose_threshold(
    prob: np.ndarray, domain: str = "auto", target_frac: float | None = None
) -> float:

    _ = target_frac

    return choose_threshold_v2(prob, domain=domain)
