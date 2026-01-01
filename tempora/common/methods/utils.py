import copy

import torch


# Recursively move all tensors in a state dict to the CPU and clone them.
def get_cpu_snapshot(state_dict):
    if isinstance(state_dict, dict):
        return {k: get_cpu_snapshot(v) for k, v in state_dict.items()}
    elif isinstance(state_dict, list):
        return [get_cpu_snapshot(v) for v in state_dict]
    elif torch.is_tensor(state_dict):
        return state_dict.detach().cpu().clone()  # Moving to CPU creates a copy; .clone() ensures it's not a view
    return copy.deepcopy(state_dict)  # For metadata like learning rate or step counts


# Source: https://github.com/davda54/sam
# Paper : https://arxiv.org/abs/2010.01412
class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}."

        super().__init__(params, defaults=dict(rho=rho, adaptive=adaptive, **kwargs))

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue

                self.state[p]["old_p"] = p.data.clone()

                # Compute perturbation and climb to the local maximum "w + e(w)"
                perturbation = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(perturbation)

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        # Get back to "w" from "w + e(w)"
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                p.data = self.state[p]["old_p"]

        self.base_optimizer.step()  # Do the actual "sharpness-aware" update

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "SAM requires a closure but wasn't provided with one."

        # The closure should do a full forward-backward pass
        closure = torch.enable_grad()(closure)

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # Shared device in case of model parallelism

        # Collect per-parameter gradient norms
        param_norms = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                # For adaptive SAM, scale gradients by parameter magnitude
                grad_for_norm = torch.abs(p) * p.grad if group["adaptive"] else p.grad
                param_norm = grad_for_norm.norm(p=2).to(shared_device)
                param_norms.append(param_norm)

        # Compute overall gradient norm as L2 norm of all parameter norms
        norm = torch.norm(torch.stack(param_norms), p=2)
        return norm