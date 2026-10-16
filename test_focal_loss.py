"""Properties focal loss has to satisfy.

Run: python test_focal_loss.py   (or: python -m pytest test_focal_loss.py)

The central check is the one the README already promises: at gamma=0 this must
equal nn.CrossEntropyLoss, weighted and unweighted. The previous version was
only ever checked against one hand-picked batch whose labels happened to make
a wrong normalisation invisible, so every case here randomises the labels.
"""
import torch
import torch.nn as nn

from focal_loss import FocalLoss


def _case(seed, n=16, c=5):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(n, c, generator=g),
            torch.randint(0, c, (n,), generator=g),
            torch.rand(c, generator=g) + 0.05)


def test_gamma_zero_equals_cross_entropy():
    for seed in range(25):
        x, y, _ = _case(seed)
        assert torch.allclose(FocalLoss(gamma=0)(x, y),
                              nn.CrossEntropyLoss()(x, y), atol=1e-6), seed


def test_gamma_zero_weighted_equals_weighted_cross_entropy():
    """Randomised labels. A single fixed batch can hide a wrong denominator."""
    for seed in range(25):
        x, y, w = _case(seed)
        assert torch.allclose(FocalLoss(weight=w, gamma=0)(x, y),
                              nn.CrossEntropyLoss(weight=w)(x, y), atol=1e-6), seed


def test_normaliser_ignores_absent_classes():
    """Dividing by sum(all class weights) instead of the batch's weight makes
    the loss depend on classes that never appear. It must not."""
    x, y, _ = _case(0, n=12, c=5)
    y = torch.zeros_like(y)                      # only class 0 is present
    w = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])
    base = FocalLoss(weight=w, gamma=2)(x, y)
    w_other = w.clone()
    w_other[3] = 1000.0                          # a class not in the batch
    assert torch.allclose(base, FocalLoss(weight=w_other, gamma=2)(x, y), atol=1e-6)


def test_reduction_modes_agree():
    x, y, w = _case(1)
    none = FocalLoss(weight=w, gamma=2, reduction="none")(x, y)
    assert none.shape == y.shape
    assert torch.allclose(none.sum(), FocalLoss(weight=w, gamma=2, reduction="sum")(x, y))
    mean = FocalLoss(weight=w, gamma=2, reduction="mean")(x, y)
    assert torch.allclose(mean, none.sum() / w.gather(0, y).sum(), atol=1e-6)


def test_ignore_index_matches_cross_entropy():
    for seed in range(10):
        x, y, w = _case(seed)
        y = y.clone()
        y[::3] = -100
        for weight in (None, w):
            got = FocalLoss(weight=weight, gamma=0, ignore_index=-100)(x, y)
            exp = nn.CrossEntropyLoss(weight=weight, ignore_index=-100)(x, y)
            assert torch.allclose(got, exp, atol=1e-6), (seed, weight is None)


def test_gamma_downweights_easy_examples():
    """The whole point: relative to cross-entropy, an easy example must lose
    more of its contribution than a hard one."""
    easy = torch.tensor([[8.0, 0.0, 0.0]])
    hard = torch.tensor([[0.4, 0.3, 0.3]])
    lab = torch.tensor([0])
    e0, h0 = FocalLoss(gamma=0)(easy, lab), FocalLoss(gamma=0)(hard, lab)
    e2, h2 = FocalLoss(gamma=2)(easy, lab), FocalLoss(gamma=2)(hard, lab)
    assert (e2 / e0) < (h2 / h0)


def test_higher_dimensional_input_matches_cross_entropy():
    """(N, C, d) inputs, as in token classification."""
    g = torch.Generator().manual_seed(7)
    x = torch.randn(4, 6, 9, generator=g)
    y = torch.randint(0, 6, (4, 9), generator=g)
    w = torch.rand(6, generator=g) + 0.05
    assert torch.allclose(FocalLoss(weight=w, gamma=0)(x, y),
                          nn.CrossEntropyLoss(weight=w)(x, y), atol=1e-6)


def test_gradients_are_finite():
    x, y, w = _case(3)
    x = x.requires_grad_(True)
    FocalLoss(weight=w, gamma=2)(x, y).backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0


def test_numerically_stable_on_extreme_logits():
    x = torch.tensor([[-1e4, 1e4, 0.0], [1e4, -1e4, 0.0]])
    y = torch.tensor([0, 1])                      # confidently wrong both times
    loss = FocalLoss(gamma=2)(x, y)
    assert torch.isfinite(loss), "overflowed on saturated logits"


def test_module_moves_with_to():
    fl = FocalLoss(weight=torch.ones(3), gamma=2)
    assert "weight" in dict(fl.named_buffers()), \
        "weight must be a buffer so .to(device) carries it"


def test_invalid_arguments_are_rejected():
    for bad in ({"gamma": -1}, {"reduction": "average"}):
        try:
            FocalLoss(**bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad} should raise ValueError")


def test_all_targets_ignored_returns_zero_not_nan():
    x, y, _ = _case(5)
    loss = FocalLoss(gamma=2, ignore_index=-100)(x, torch.full_like(y, -100))
    assert torch.isfinite(loss) and float(loss) == 0.0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = []
    for fn in fns:
        try:
            fn()
            print(f"  ok    {fn.__name__}")
        except Exception as e:
            failed.append(fn.__name__)
            msg = (str(e).splitlines() or [type(e).__name__])[0]
            print(f"  FAIL  {fn.__name__}: {msg[:88]}")
    print(f"\n{len(fns)-len(failed)} passed, {len(failed)} failed")
    raise SystemExit(1 if failed else 0)
