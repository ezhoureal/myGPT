import torch
from llm_core.optimizer import CustomAdamOptimizer


def _optimize(opt_class) -> torch.Tensor:
    '''
    This test function is borrowed from https://github.com/stanford-cs336/assignment1-basics/blob/main/tests/test_optimizer.py
    '''
    torch.manual_seed(55)
    model = torch.nn.Linear(3, 2, bias=False)
    opt = opt_class(
        model.parameters(),
        lr=1e-3,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    for _ in range(1000):
        opt.zero_grad()
        x = torch.rand(model.in_features)
        y_hat = model(x)
        y = torch.tensor([x[0] + x[1], -x[2]])
        loss = ((y - y_hat) ** 2).sum()
        loss.backward()
        opt.step()
    return model.weight.detach()


def test_adamw():
    pytorch_weights = _optimize(torch.optim.AdamW)
    actual_weights = _optimize(CustomAdamOptimizer)
    matches_pytorch = torch.allclose(
        actual_weights, pytorch_weights, atol=1e-4)
    assert (matches_pytorch)
