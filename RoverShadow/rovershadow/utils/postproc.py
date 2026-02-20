from __future__ import annotations

import numpy as np

try:

    import cv2

except Exception as e:

    cv2 = None


def remove_small_components(mask01: np.ndarray, min_area: int) -> np.ndarray:
    """
    Remove connected foreground components smaller than min_area.
    mask01: HxW uint8 {0,1}
    returns: HxW uint8 {0,1}
    """

    if min_area is None or int(min_area) <= 0:

        return mask01

    if cv2 is None:

        raise RuntimeError(
            "cv2 not available but remove_small_components() was called."
        )

    m = mask01.astype(np.uint8) * 255

    if m.max() == 0:

        return mask01

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)

    if num_labels <= 1:

        return mask01

    areas = stats[:, cv2.CC_STAT_AREA]

    keep = areas >= int(min_area)

    keep[0] = True

    out = keep[labels].astype(np.uint8)

    return out


def morph_close(mask01: np.ndarray, k: int) -> np.ndarray:
    """
    Morphological closing (dilate then erode) with kxk kernel.
    mask01: HxW uint8 {0,1}
    returns: HxW uint8 {0,1}
    """

    if k is None or int(k) <= 0:

        return mask01

    if cv2 is None:

        raise RuntimeError("cv2 not available but morph_close() was called.")

    kk = int(k)

    kernel = np.ones((kk, kk), np.uint8)

    m = mask01.astype(np.uint8) * 255

    out = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)

    return (out > 0).astype(np.uint8)


def apply_postproc(
    mask01: np.ndarray, min_area: int = 0, close_k: int = 0
) -> np.ndarray:
    """
    Apply post-processing in stable order:
      1) close holes / connect fragments
      2) remove tiny blobs
    """

    out = mask01.astype(np.uint8)

    if int(close_k) > 0:

        out = morph_close(out, int(close_k))

    if int(min_area) > 0:

        out = remove_small_components(out, int(min_area))

    return out
