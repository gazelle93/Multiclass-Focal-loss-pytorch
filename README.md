# Overview
- This is an implementation of multi-class focal loss in PyTorch.

# Brief description
- This loss function generalizes multiclass cross-entropy by introducing a hyperparameter gamma(focusing parameter) that allows to focus on hard examples. The Focal loss: $FL(p_t)=-\alpha_t(1-p_t)^\gamma\log(p_t)$, where $\alpha_t$ is a weighting facrot, $p_t$ is a model's estimated probability, $\gamma$ is a focusing parameter and $-\log(p_t)$ is the cross entropy loss in this case.

# Prerequisites
> - torch

# Usage
```python
from focal_loss import FocalLoss

criterion = FocalLoss(weight=class_weights, gamma=2.0)
loss = criterion(logits, targets)      # logits (N, C) or (N, C, d1, ...)
```
It is a drop-in replacement for `nn.CrossEntropyLoss`: same `weight`,
`reduction` and `ignore_index` semantics, and identical output at `gamma=0`.

# Parameters
> - gamma(float, defaults to 2.0): The focusing parameter (must be non-negative). `gamma=0` reduces to cross-entropy.
> - weight(Tensor, Optional): Weighting factor for each of the n classes, acting as alpha_t.
> - reduction(str, defaults to "mean"): One of "mean", "sum" or "none". With weights, "mean" divides by the summed weight of the batch, matching `nn.CrossEntropyLoss`.
> - ignore_index(int, defaults to -100): Target value to skip entirely.

# References
- Focal loss: Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal loss for dense object detection. In Proceedings of the IEEE international conference on computer vision (pp. 2980-2988).

# Corrections (2026-10-16)

An audit found that this code did not run and, where it did, computed the
wrong number when class weights were used. If you used it, the second item is
the one worth re-checking.

- **The package did not run.** `main.py` imported the module and then called
  it (`import focal_loss as Focalloss` followed by `Focalloss(...)`), so the
  only entry point raised `TypeError` immediately.
- **The unweighted path always crashed.** The no-weight branch divided by
  `len(probs)`, a name that only existed in `main.py`, so the default
  `FocalLoss()` raised `NameError` on every call.
- **Weighted loss used the wrong denominator.** It divided by the sum of all
  class weights rather than the summed weight of the samples in the batch,
  which is what `nn.CrossEntropyLoss` does. The example in this repository
  happened to hide it: with weights `[1, 200, 1e-5]` and labels `[2, 2, 0, 1]`
  the two denominators agree to five decimal places. With labels `[1, 1, 1, 1]`
  the loss came out 3.98x too large. Every check now randomises the labels.
- **The forward pass looped over the batch in Python**, calling
  `nn.CrossEntropyLoss` once per sample. It is now vectorised: 13x faster at a
  batch of 256 and 149x at 2048.
- **No `reduction`, no `ignore_index`.** Both are now supported with the same
  semantics as `nn.CrossEntropyLoss`, along with `(N, C, d1, ...)` inputs for
  token-level tasks.
- **The `device` argument is gone.** `weight` is a registered buffer, so the
  standard `.to(device)` already handles it; the extra argument only moved the
  output, which was already on the right device.

`test_focal_loss.py` pins all of this. Against the previous code 10 of its 12
checks fail.
