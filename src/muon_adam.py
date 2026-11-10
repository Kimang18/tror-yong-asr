import math

import torch

polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


@torch.compile
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power/orthogonalization of G.
    Follow nanogpt repo implementation.
    """
    assert G.ndim >= 2
    X = G.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)
    if G.size(-2) > G.size(-1):  # Tall matrix
        for a, b, c in polar_express_coeffs[:steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:  # Wide matrix (original math)
        for a, b, c in polar_express_coeffs[:steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    return X


class MuonWithAuxAdam(torch.optim.Optimizer):
    """
    https://github.com/KellerJordan/Muon/blob/6399c658d3c4a3356ba823fa6664b10e23871068/muon.py#L228
    """

    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["params"] = sorted(
                    group["params"], key=lambda x: x.size(), reverse=True
                )
                # Muon config
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0.1)
                group["nesterov"] = group.get("nesterov", True)
                group["steps"] = group.get("steps", 5)
                assert set(group.keys()) == set(
                    [
                        "params",
                        "lr",
                        "momentum",
                        "weight_decay",
                        "use_muon",
                        "steps",
                        "nesterov",
                    ]
                )
            else:
                # Adam config
                group["lr"] = group.get("lr", 7e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-6)
                group["weight_decay"] = group.get("weight_decay", 0.1)
                assert set(group.keys()) == set(
                    ["params", "lr", "betas", "eps", "weight_decay", "use_muon"]
                )
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                lr = group["lr"]
                wd = group["weight_decay"]
                beta = group["momentum"]
                nesterov = group["nesterov"]
                for p in group["params"]:
                    g = p.grad
                    if g is None:
                        continue

                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.lerp_(g, 1 - beta)
                    u = buf.lerp(g, 1 - beta) if nesterov else buf
                    if u.ndim == 4:  # for the case of conv filters
                        u = u.view(len(u), -1)
                    u = zeropower_via_newtonschulz5(u, steps=group["steps"])

                    scale = 0.2 * math.sqrt(max(u.size(-2), u.size(-1)))

                    # apply weight decay
                    p.data.mul_(1 - lr * wd)
                    # apply update
                    p.data.add_(u, alpha=-lr * scale)

            else:
                lr = group["lr"]
                wd = group["weight_decay"]
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                for p in group["params"]:
                    g = p.grad
                    if g is None:
                        continue

                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(g)
                        state["exp_avg_sq"] = torch.zeros_like(g)
                        state["step"] = 0
                    state["step"] += 1
                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]
                    exp_avg.lerp_(g, 1 - beta1)
                    exp_avg_sq.lerp_(g.square(), 1 - beta2)
                    u = exp_avg / (eps + exp_avg_sq.sqrt())

                    step = state["step"]
                    scale = math.sqrt(1 - beta2**step) / (1 - beta1**step)

                    # apply weight decay
                    p.data.mul_(1 - lr * wd)
                    # apply update
                    p.data.add_(u, alpha=-lr * scale)
        return loss
