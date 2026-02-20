from __future__ import annotations

import numpy as np

from rovershadow.utils.postproc import apply_postproc


def prob_to_mask01(
    prob: np.ndarray,
    threshold: float,
    min_area: int = 0,
    close_k: int = 0,
) -> np.ndarray:
    """
    Convert a shadow probability map into a {0,1} uint8 mask.

    prob: HxW float32 in [0,1]
    threshold: scalar threshold
    min_area: remove connected components smaller than this (0 disables)
    close_k: morphological closing kernel size (0 disables)

    returns: HxW uint8 in {0,1}
    """

    mask01 = (prob >= float(threshold)).astype(np.uint8)

    mask01 = apply_postproc(mask01, min_area=int(min_area), close_k=int(close_k))

    return mask01
