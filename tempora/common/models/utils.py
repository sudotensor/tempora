import torch.nn as nn


# Source: https://github.com/mr-eggplant/SPA/blob/main/models/byol_wrapper.py
# BYOL-inspired wrapper that adds a learnable projector before the classification head.
# Note: Used for SPA's weak augmented view predictions to prevent collapse.
class FeatureAdapter(nn.Module):
    def __init__(self, arch, model, projector_dim):
        super().__init__()
        self.arch = arch
        self.model = model
        self.projector = nn.Linear(projector_dim, projector_dim, bias=False)
        nn.init.eye_(self.projector.weight)

    # When using the projector, features returned are ones after the projection layer.
    # Note: The feature returning functionality is only maintained for API consistency.
    def forward(self, x, use_projector=False, return_feature_only=False, return_feature_and_logits=False):
        if not use_projector:
            return self.model(x, return_feature_only, return_feature_and_logits)

        x = self.model(x, return_feature_only=True)
        x = self.projector(x)

        if return_feature_only:
            return x
        
        feature = x if return_feature_and_logits else None

        match self.arch:
            case "resnet18" | "resnet50" | "resnet50_gn":
                x = self.model.fc(x)
            case "mobilenet_v2" | "mobilenet_v3_small":
                x = self.model.classifier(x)
            case "vit_base_patch16_224":
                x = self.model.head(x)
            case _:
                raise ValueError(f"Unsupported architecture: {self.arch}")
            
        if return_feature_and_logits:
            return x, feature

        return x