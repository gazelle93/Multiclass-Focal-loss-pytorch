import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.modules.loss._WeightedLoss):
    """Multi-class focal loss, Lin et al. (2017).

        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Drop-in for nn.CrossEntropyLoss: same `weight`, `ignore_index` and
    `reduction` semantics, and identical output at gamma=0.

    Args:
        weight: per-class weights acting as alpha_t, shape (C,).
        gamma: focusing parameter, >= 0. gamma=0 reduces to cross-entropy.
        reduction: 'mean', 'sum' or 'none'. As in nn.CrossEntropyLoss,
            'mean' with weights divides by the summed weight of the batch,
            not by the number of samples.
        ignore_index: target value to skip entirely.
    """

    def __init__(self, weight=None, gamma=2.0, reduction="mean",
                 ignore_index=-100):
        super(FocalLoss, self).__init__(weight, reduction=reduction)
        if gamma < 0:
            raise ValueError(f"gamma must be non-negative, got {gamma}")
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(
                f"reduction must be 'mean', 'sum' or 'none', got {reduction!r}")
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, _input, _target):
        if _input.dim() < 2:
            raise ValueError(
                f"expected input of shape (N, C, ...), got {tuple(_input.shape)}")

        # (N, C, d1, d2, ...) -> (N * d1 * d2 * ..., C), matching how
        # nn.CrossEntropyLoss flattens higher-dimensional inputs
        num_classes = _input.size(1)
        flat_input = _input.transpose(0, 1).reshape(num_classes, -1).t()
        flat_target = _target.reshape(-1)

        keep = flat_target != self.ignore_index
        if not bool(keep.all()):
            flat_input, flat_target = flat_input[keep], flat_target[keep]

        if flat_target.numel() == 0:
            zero = _input.sum() * 0.0
            return zero if self.reduction != "none" else zero.reshape(0)

        # -log(p_t), one per sample. log_softmax keeps this stable.
        log_pt = F.log_softmax(flat_input, dim=-1).gather(
            1, flat_target.unsqueeze(1)).squeeze(1)
        ce = -log_pt
        pt = log_pt.exp()

        loss = (1.0 - pt) ** self.gamma * ce

        if self.weight is not None:
            alpha = self.weight.to(dtype=loss.dtype, device=loss.device)
            sample_weight = alpha.gather(0, flat_target)
            loss = loss * sample_weight

        if self.reduction == "none":
            return loss
        if self.reduction == "sum":
            return loss.sum()
        # 'mean': weighted mean normalises by the weight present in the batch,
        # which is what nn.CrossEntropyLoss does. Dividing by the sum over all
        # classes instead makes the result depend on classes that never appear.
        if self.weight is not None:
            return loss.sum() / sample_weight.sum()
        return loss.mean()
