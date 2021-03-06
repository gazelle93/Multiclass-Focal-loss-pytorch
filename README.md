# Overview
- This is an implementation of multi-class focal loss in PyTorch.

# Brief description
- This loss function generalizes multiclass cross-entropy by introducing a hyperparameter gamma(focusing parameter) that allows to focus on hard examples. The Focal loss: $FL(p_t)=-\alpha_t(1-p_t)^\gamma\log(p_t)$, where $\alpha_t$ is a weighting facrot, $p_t$ is a model's estimated probability, $\gamma$ is a focusing parameter and $-\log(p_t)$ is the cross entropy loss in this case.

# Prerequisites
> - torch

# Parameters
> - gamma(int): The focusing parameter (Must be non-negative).
> - weight(Tensor, Optional): Weighting factor for each of the n classes.

# References
- Focal loss: Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal loss for dense object detection. In Proceedings of the IEEE international conference on computer vision (pp. 2980-2988).
