import math
from typing import Callable, Optional
import torch
class CustomAdamOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, beta_1=0.99, beta_2=0.999, epsilon=1e-8, decay=0.01):
        if lr < 0:
            raise ValueError(f'invalid learning rate {lr}')
        defaults = {"lr": lr, "beta_1": beta_1, "beta_2": beta_2, "epsilon": epsilon, "decay": decay}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            beta1 = group["beta_1"]
            beta2 = group["beta_2"]
            epsilon = group["epsilon"]
            decay = group["decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] # Get state associated with p.
                state["t"] = state.get("t", 0) + 1 # Increment iteration number.

                grad = p.grad.data # Get the gradient of loss with respect to p.
                state["m"] = state.get("m", 0) * beta1 + grad * (1 - beta1) # first moment estimate
                state["v"] = state.get("v", 0) * beta2 + grad ** 2 * (1 - beta2) # second moment estimate

                alpha_t = lr * math.sqrt(1 - state["v"] ** state["t"]) / (1 - state["m"] ** state["t"])
                p.data -= alpha_t * state["m"] / (math.sqrt(state["v"]) + epsilon) # Update weight tensor in-place.
                p.data -= decay * lr * p.data # apply weight decay
        return loss
