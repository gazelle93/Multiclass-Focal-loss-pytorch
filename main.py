import torch
import torch.nn as nn

from focal_loss import FocalLoss

device = "cuda" if torch.cuda.is_available() else "cpu"

# a deliberately lopsided weighting: class 1 matters, class 2 barely counts
weight = torch.FloatTensor([1, 200, 1e-5]).to(device)

logits = torch.tensor([[0.0, 0.2, 0.8],
                       [0.0, 0.2, 0.5],
                       [0.8, 0.5, 0.1],
                       [0.2, 0.5, 0.3]]).to(device)

labels = torch.tensor([2, 2, 0, 1]).to(device)

print("Focal Loss (gamma=2, weighted): {:.6f}\n".format(
    FocalLoss(weight=weight, gamma=2).to(device)(logits, labels)))


# gamma=0 removes the focusing term, so focal loss collapses to cross-entropy.
# Checked on random labels rather than one fixed batch: the denominator bug
# this guards against was invisible for the batch above and wrong by ~4x for
# others, so a single example proves nothing.
print("gamma=0 must equal cross-entropy, weighted and unweighted.")
for trial in range(5):
    g = torch.Generator().manual_seed(trial)
    x = torch.randn(32, 3, generator=g).to(device)
    y = torch.randint(0, 3, (32,), generator=g).to(device)

    plain = FocalLoss(gamma=0).to(device)(x, y)
    weighted = FocalLoss(weight=weight, gamma=0).to(device)(x, y)
    ce = nn.CrossEntropyLoss()(x, y)
    wce = nn.CrossEntropyLoss(weight=weight)(x, y)

    assert torch.allclose(plain, ce, atol=1e-6)
    assert torch.allclose(weighted, wce, atol=1e-6)
    print(f"  trial {trial}: focal {plain:.6f} = CE {ce:.6f}   "
          f"weighted {weighted:.6f} = weighted CE {wce:.6f}")

print("\nAll checks passed. See test_focal_loss.py for the full set.")
