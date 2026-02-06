import timm


def get_vit(arch: str, dataset: str, checkpoint: str | None = None, device: str = "cpu"):
    """Get a ViT model for the specified architecture variant and dataset

    Args:
        arch:       'vit_base_patch16_224'
        dataset:    'imagenet'
        checkpoint: Unsupported, must be None
        device:     Device to load model on
    """

    if arch not in ["vit_base_patch16_224"]:
        raise ValueError(f"Unsupported architecture: {arch}. Choose 'vit_base_patch16_224'.")
    if dataset != "imagenet":
        raise ValueError(f"Unsupported dataset: '{dataset}'. Choose 'imagenet'.")
    if checkpoint is not None:
        raise ValueError("Custom checkpoints are not supported for 'vit_base_patch16_224'.")

    model = timm.create_model(arch, num_classes=1000, pretrained=True)
    model = _patch_vit_timm(model)
    model.to(device)
    model.eval()

    return model


# Monkeypatch a ViT model (from Timm) to return features along with predictions if requested.
# Note: In the timm library, forward_head supports dropout; we don't. 
def _patch_vit_timm(model):
    def forward(self, x, return_feature_only=False, return_feature_and_logits=False):
        x = self.forward_features(x)
        x = self.pool(x)
        x = self.fc_norm(x)

        if return_feature_only:
            return x

        feature = x if return_feature_and_logits else None
        x = self.head(x)

        if return_feature_and_logits:
            return x, feature

        return x

    model.forward = forward.__get__(model, model.__class__)

    return model