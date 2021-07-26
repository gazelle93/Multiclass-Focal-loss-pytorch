import torch
import torch.nn as nn
import focal_loss as Focalloss

weight = torch.FloatTensor([1, 200, 1e-5])

probs = torch.tensor([[0.0, 0.2, 0.8],
                       [0.0, 0.2, 0.5],
                       [0.8, 0.5, 0.1],
                       [0.2, 0.5, 0.3]])

labels = torch.tensor([2,2,0,1])


focal_loss = Focalloss(weight=weight, gamma=2)
print("Focal Loss: {}\n".format(focal_loss(probs, labels)))


"""Focal loss with gamma=0 should be the same as cross-entropy."""
print("Focal loss with gamma=0 and no weight should be the same as cross-entropy.")
ce_loss = nn.CrossEntropyLoss()
print("Cross-Entropy Loss: {}".format(ce_loss(probs, labels)))

focal_loss = Focalloss(weight=None, gamma=0)
print("Focal Loss(gamma=0, no weight): {}\n".format(focal_loss(probs, labels)))


"""Focal loss with gamma=0 & weight should be the same as weighted cross-entropy."""
print("Focal loss with gamma=0 and weight should be the same as cross-entropy.")
wce_loss = nn.CrossEntropyLoss(weight=weight)
print("Weighted Cross-Entropy Loss: {}".format(wce_loss(probs, labels)))

focal_loss = Focalloss(weight=weight, gamma=0)
print("Focal Loss(gamma=0): {}".format(focal_loss(probs, labels)))
