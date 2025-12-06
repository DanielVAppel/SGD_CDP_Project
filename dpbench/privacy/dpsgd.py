from opacus import PrivacyEngine

def attach_privacy(model, optimizer, dataloader, noise_multiplier, max_grad_norm, target_delta):
    privacy_engine = PrivacyEngine()
    model, optimizer, dataloader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=dataloader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
    )
    return privacy_engine, model, optimizer, dataloader

def get_epsilon(privacy_engine, delta):
    try:
        return privacy_engine.get_epsilon(delta)
    except Exception:
        return None
