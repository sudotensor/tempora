import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import stopwatch
from .base import Method
from .utils import get_cpu_snapshot


# Source: https://github.com/mr-eggplant/SPA/tree/main
# Paper : https://openreview.net/forum?id=Li4rieeClO
# Note  : This version adapts a model using LF amplitude masking and HF noise injection with consistency loss.
class SPA(Method):
    def __init__(self, model, optimizer, reforward=False, n_ratio=0.4, m_ratio=0.2, combined=False):
        super().__init__()
        self.frozen = False

        self.model = self._configure_model(model)
        self.optimizer = optimizer
        self.reforward = reforward

        self.n_ratio = n_ratio
        self.m_ratio = m_ratio

        self.combined = combined

        self.lf_aug = LFAugment(alpha=0.2, image_size=224)
        self.hf_aug = HFAugment(n_ratio=n_ratio, image_size=224, patch_size=16)
        self.hf_optimizer = None  # Created in forward when device is known

        self.model_state = get_cpu_snapshot(self.model.state_dict())
        self.optimizer_state = get_cpu_snapshot(self.optimizer.state_dict())
        self.hf_aug_state = get_cpu_snapshot(self.hf_aug.state_dict())

        self._check_model(self.model)

    def forward(self, x, device):
        def _forward():
            outputs, prediction_time = None, 0.0

            self.lf_aug.to(device)
            self.hf_aug.to(device)

            if self.hf_optimizer is None:
                self.hf_optimizer = torch.optim.SGD([self.hf_aug.alpha], lr=1, momentum=0.9)

            with torch.enable_grad():
                outputs, prediction_time = stopwatch(device, lambda: self.model(x))  # Strong view

                if self.combined:  # SPA-I
                    x1 = self.hf_aug(self.lf_aug(x, m_ratio=self.m_ratio))  # Both augmentations on a single image
                    loss = self._get_consistency_loss(x1, outputs)
                else:
                    # Structure-preserving augmented (weak) views (Eqns. 4-5)
                    x1 = self.lf_aug(x, m_ratio=self.m_ratio)
                    x2 = self.hf_aug(x)

                    # Active loss calculation (Eqn. 1)
                    loss = self._get_consistency_loss(x1, outputs) + self._get_consistency_loss(x2, outputs)

                loss.backward()

                # Negate gradient for HF augment to maximize perturbation
                if self.hf_aug.alpha.grad is not None:
                    self.hf_aug.alpha.grad.neg_()

                self.optimizer.step()
                self.optimizer.zero_grad()

                self.hf_optimizer.step()
                self.hf_optimizer.zero_grad()

            # Renormalise alphas to maintain target average noise ratio
            with torch.no_grad():
                mean_ratio = (self.hf_aug.alpha ** 2).mean()
                self.hf_aug.alpha.data.div_(torch.sqrt(mean_ratio / self.n_ratio))
                self.hf_aug.alpha.data.clamp_(-1, 1)

            return outputs, prediction_time

        if self.frozen:
            with torch.no_grad():
                return stopwatch(device, lambda: self.model(x).softmax(1))

        (outputs, prediction_time), adaptation_time = stopwatch(device, _forward)

        if self.reforward:
            with torch.no_grad():
                self.freeze()
                outputs, reforward_time = stopwatch(device, lambda: self.model(x))
                self.unfreeze()
                prediction_time = adaptation_time + reforward_time

        return outputs.softmax(1), prediction_time

    def _get_consistency_loss(self, x, s_output):
        w_output = self.model(x, use_projector=True)

        s_confidence, _ = s_output.softmax(dim=-1).max(dim=-1)
        w_confidence, _ = w_output.softmax(dim=-1).max(dim=-1) 

        # Only use samples where the strong view is more confident than the weak view (Eqn. 2)
        sample_filter = torch.where(s_confidence > w_confidence)
        y_prd = w_output[sample_filter].log_softmax(dim=-1)
        y_tgt = s_output[sample_filter].softmax(dim=-1).detach()

        return F.kl_div(y_prd, y_tgt, reduction="batchmean")

    def reset(self):
        self.model.load_state_dict(self.model_state, strict=True)
        self.optimizer.load_state_dict(self.optimizer_state)
        self.hf_aug.load_state_dict(self.hf_aug_state, strict=True)  # type:ignore
        self.hf_optimizer = None

    def freeze(self):
        self.frozen = True
        self.model.eval()

    def unfreeze(self):
        self.frozen = False
        self.model.train()

    @staticmethod
    def collect_params(model):
        ps = []
        ns = []
        for nm, m in model.named_modules():
            if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
                for np, p in m.named_parameters():
                    if np in ["weight", "bias"]:
                        ps.append(p)
                        ns.append(f"{nm}.{np}")
        return ps, ns

    def _configure_model(self, model):
        model.train()
        model.requires_grad_(False)

        if hasattr(model, "projector"):
            model.projector.requires_grad_(True)

        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                m.reset_running_stats()  # Discard source statistics
            elif isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
                m.requires_grad_(True)

        return model

    def _check_model(self, model):
        assert model.training, "SPA requires train mode."

        param_grads = [p.requires_grad for p in model.parameters()]
        assert any(param_grads), "SPA needs some parameters with gradients."
        assert not all(param_grads), "SPA should not update all parameters."

        has_norm = any(isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)) for m in model.modules())
        assert has_norm, "SPA requires normalization layers."


# Low-frequency augmentation via amplitude masking in the Fourier domain (Eqn. 4)
# Note: Alpha controls the size of a square placed at the center of the transformed image that defines the low-frequency
#       region. The mask ratio is the proportion of low-frequency points within this region to zero out. The paper sets 
#       both to 0.2. They just happen to share the same default value but control different things.
#       FFT -> shift -> mask center -> unshift -> IFFT
class LFAugment(nn.Module):
    def __init__(self, alpha=0.2, image_size=224):
        super().__init__()
        region_size = max(1, int(image_size * alpha))
        border_size = (image_size - region_size) // 2

        ones = torch.ones((1, 3, region_size, region_size))
        mask = F.pad(ones, [border_size, image_size - region_size - border_size] * 2, value=0.0).contiguous()
      
        self.register_buffer("l_mask", mask)
        self.register_buffer("h_mask", 1 - mask)

    def forward(self, img, m_ratio=0.1):
        fft = torch.fft.fft2(img.clone(), dim=(-2, -1))
        amp, phase = torch.abs(fft), torch.angle(fft)

        # Mask low-frequency amplitudes (center region after fftshift)
        amp = torch.fft.fftshift(amp)
        amp = self._apply_mask(amp, keep_prob=1 - m_ratio)
        amp = torch.fft.ifftshift(amp)

        return torch.fft.ifft2(torch.polar(amp, phase), dim=(-2, -1)).real  # Reconstruct image

    # Randomly zero out low-frequency points in the mask region.
    # Note: _make_symmetric enforces that zeros come in symmetric pairs, which is required for IFFT to produce a real
    #       output. This doesn't change the effective mask ratio. 
    def _apply_mask(self, amp, keep_prob):
        mask = self.l_mask * torch.empty_like(amp).bernoulli_(keep_prob)  # type:ignore
        mask = mask + self.h_mask  # type:ignore
        mask = self._make_symmetric(mask)
        return amp * mask

    # FFT of real images has conjugate symmetry. If we mask position k, we must also mask -k, otherwise IFFT produces
    # complex output. The top half of the mask acts as a source, with the bottom half serving as its mirror. The mask is
    # temporarily adjusted to extract an odd-sized region so that there's a true center point for symmetry; the skipped 
    # row/col are in the border, so that does not challenge the symmetry.
    def _make_symmetric(self, mask):
        trim_idx = 1 - (mask.shape[-2] % 2)  # 0 if even, 1 if odd
        sub_mask = mask[:, :, trim_idx:, trim_idx:]
        half_idx = sub_mask.shape[-2] // 2

        # Set the bottom half and left side of the middle row to be 1 
        sub_mask[:, :, -half_idx:, :] = 1
        sub_mask[:, :, half_idx, :half_idx] = 1

        # Multiply by its 180-degree flip to enforce symmetry.
        mask[:, :, trim_idx:, trim_idx:] = sub_mask * torch.flip(sub_mask, dims=[-1, -2])

        return mask


# High-frequency augmentation via learnable patch-wise noise injection (Eqn. 5)
# Note: Divides the image into patches and blends each with Gaussian noise. The blend strength is learnable per-patch
#       and squared to keep it non-negative. The default ratio of 0.4 indicates 40% noise. 
class HFAugment(nn.Module):
    def __init__(self, n_ratio=0.4, image_size=224, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        self.alpha = nn.Parameter(torch.ones((image_size // patch_size) ** 2) * math.sqrt(n_ratio), requires_grad=True)

    def forward(self, img):
        assert img.shape[-1] == img.shape[-2] and img.shape[-1] % self.patch_size == 0

        n = img.shape[-1] // self.patch_size  # Number of patches per row/column

        aug_img = torch.empty_like(img)
        origins = [(i * self.patch_size, j * self.patch_size) for i in range(n) for j in range(n)]  # Top-left corners
        for idx, (y, x) in enumerate(origins):
            patch = img[..., y : y + self.patch_size, x : x + self.patch_size]
            noise = torch.randn(*patch.shape[1:], device=img.device)
            ratio = self.alpha[idx] ** 2
            
            aug_img[..., y : y + self.patch_size, x : x + self.patch_size] = (1 - ratio) * patch + ratio * noise
        
        return aug_img.clip_(-1, 1)
