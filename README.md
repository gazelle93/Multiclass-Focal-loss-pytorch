# Overview
- This is an implementation of multi-class focal loss in pytorch.

# Brief description
- This loss function generalizes multiclass cross-entropy by introducing a hyperparameter gamma(focusing parameter) that allows to focus on hard examples. The Focal loss: <img src="https://render.githubusercontent.com/render/math?math=FL(p_t)=-\alpha_t(1-p_t)^\gamma\log(p_t)">, where <img src="https://render.githubusercontent.com/render/math?math=\alpha"> is a weighting facrot, <img src="https://render.githubusercontent.com/render/math?math=p_t"> is a model's estimated probability, <img src="https://render.githubusercontent.com/render/math?math=\gamma"> is a focusing parameter and <img src="https://render.githubusercontent.com/render/math?math=-\log(p_t)"> is  the cross entropy loss.

# Prerequisites
> - torch

# Parameters
> - gamma(int): The focusing parameter (Must be non-negative).
> - weight(Tensor, Optional): Weighting factor for each of the n classes.

# References
- Focal loss: Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal loss for dense object detection. In Proceedings of the IEEE international conference on computer vision (pp. 2980-2988).
