from __future__ import annotations

import torch

import torch.nn as nn

import torch.nn.functional as F

from mmseg.registry import MODELS


@MODELS.register_module()
class SafeCrossEntropyLoss(nn.Module):

    def __init__(
        self,
        use_sigmoid: bool = False,
        class_weight=None,
        reduction: str = "mean",
        loss_weight: float = 1.0,
        avg_non_ignore: bool = True,
        loss_name: str = "loss_ce",
        **kwargs,
    ) -> None:

        super().__init__()

        self.reduction = reduction

        self.loss_weight = loss_weight

        self.avg_non_ignore = avg_non_ignore

        self._loss_name = loss_name

        if class_weight is not None:

            self.register_buffer(
                "_class_weight",
                torch.tensor(class_weight, dtype=torch.float32),
                persistent=False,
            )

        else:

            self._class_weight = None

    def forward(
        self,
        cls_score,
        label,
        weight=None,
        avg_factor=None,
        reduction_override=None,
        ignore_index=255,
        **kwargs,
    ):

        reduction = reduction_override or self.reduction

        class_weight = (
            self._class_weight.to(cls_score.device)
            if self._class_weight is not None
            else None
        )

        loss = F.cross_entropy(
            cls_score,
            label.long(),
            weight=class_weight,
            ignore_index=ignore_index,
            reduction="none",
        )

        if weight is not None and weight.shape == loss.shape:

            loss = loss * weight

        valid = label != ignore_index

        if reduction == "none":

            out = loss

        elif reduction == "sum":

            out = loss[valid].sum() if valid.any() else (loss.sum() * 0.0)

        else:

            denom = (
                float(avg_factor)
                if avg_factor is not None
                else max(int(valid.sum().item()), 1)
            )

            out = (loss[valid].sum() / denom) if valid.any() else (loss.sum() * 0.0)

        return out * self.loss_weight

    @property
    def loss_name(self):

        return self._loss_name
