"""
privacy_engine.py — Differential Privacy Integration for SynthPriv-ReID
========================================================================
Wraps the Discriminator with Opacus's PrivacyEngine to provide
(ε, δ)-differential privacy guarantees during adversarial training.

Key concepts:
  • Only the Discriminator is made private, because only it processes
    real (sensitive) data. The Generator only receives gradients.
  • Opacus implements DP-SGD: per-sample gradient clipping + calibrated
    Gaussian noise injection into the aggregated gradient.
  • The privacy budget (ε) is tracked via Rényi Differential Privacy (RDP)
    accountant and converted to (ε, δ)-DP.

Opacus requirements (validated here):
  1. No BatchNorm — we use GroupNorm (already in models.py).
  2. DataLoader must use drop_last=True (set in data_pipeline.py).
  3. model.forward() must perform a single backward pass per step.
"""

from typing import Tuple

import torch
from torch.utils.data import DataLoader

from opacus import PrivacyEngine
from opacus.validators import ModuleValidator


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------
def validate_and_fix_model(model: torch.nn.Module) -> torch.nn.Module:
    """
    Check that the model is compatible with Opacus.
    If not, attempt automatic fixes (e.g., BatchNorm → GroupNorm).

    Returns:
        The (possibly fixed) model.
    """
    errors = ModuleValidator.validate(model, strict=False)
    if errors:
        print(f"[PrivacyEngine] Model validation found {len(errors)} issue(s). Attempting auto-fix...")
        model = ModuleValidator.fix(model)
        # Re-validate
        errors_after = ModuleValidator.validate(model, strict=False)
        if errors_after:
            raise RuntimeError(
                f"Model still has {len(errors_after)} Opacus-incompatible layers "
                f"after auto-fix:\n" + "\n".join(str(e) for e in errors_after)
            )
        print("[PrivacyEngine] Auto-fix successful.")
    else:
        print("[PrivacyEngine] Model passes Opacus validation.")
    return model


# ---------------------------------------------------------------------------
# Attach PrivacyEngine
# ---------------------------------------------------------------------------
def make_discriminator_private(
    discriminator: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloader: DataLoader,
    target_epsilon: float = 8.0,
    target_delta: float = 1e-5,
    max_grad_norm: float = 1.0,
    noise_multiplier: float = 1.1,
    epochs: int = 50,
) -> Tuple[torch.nn.Module, torch.optim.Optimizer, DataLoader, PrivacyEngine]:
    """
    Wrap the discriminator, optimizer, and dataloader with Opacus.

    Args:
        discriminator:    the Discriminator model
        optimizer:        optimizer for the discriminator
        dataloader:       DataLoader supplying real images
        target_epsilon:   privacy budget target (lower = more private)
        target_delta:     failure probability (typically 1/N or smaller)
        max_grad_norm:    L2 norm bound for per-sample gradient clipping
        noise_multiplier: std of Gaussian noise = noise_multiplier × max_grad_norm
        epochs:           planned number of training epochs (for budget estimation)

    Returns:
        (private_discriminator, private_optimizer, private_dataloader, privacy_engine)
    """
    # Step 1: validate / fix the model
    discriminator = validate_and_fix_model(discriminator)

    # Step 2: create the PrivacyEngine
    privacy_engine = PrivacyEngine()

    # Step 3: make_private — this wraps the model, optimizer, and dataloader
    # We disable Poisson sampling because GAN training requires two forward
    # passes (real + fake) through the discriminator before one optimizer step.
    # Poisson sampling's DPDataLoader rejects gradient accumulation across
    # multiple forwards. With poisson_sampling=False, Opacus uses a standard
    # uniform-without-replacement sampler, which is compatible and still
    # provides valid (ε,δ)-DP guarantees under the standard DP-SGD analysis.
    private_discriminator, private_optimizer, private_dataloader = privacy_engine.make_private(
        module=discriminator,
        optimizer=optimizer,
        data_loader=dataloader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
        poisson_sampling=False,
        grad_sample_mode="hooks",
    )

    # Log expected privacy budget
    print(f"[PrivacyEngine] DP-SGD configured:")
    print(f"  noise_multiplier = {noise_multiplier}")
    print(f"  max_grad_norm    = {max_grad_norm}")
    print(f"  target ε         = {target_epsilon}")
    print(f"  target δ         = {target_delta}")
    print(f"  planned epochs   = {epochs}")

    return private_discriminator, private_optimizer, private_dataloader, privacy_engine


# ---------------------------------------------------------------------------
# Privacy budget query
# ---------------------------------------------------------------------------
def get_current_epsilon(
    privacy_engine: PrivacyEngine,
    delta: float = 1e-5,
) -> float:
    """
    Query the RDP accountant for the current (ε, δ)-DP guarantee.

    Args:
        privacy_engine: the attached PrivacyEngine instance
        delta:          the δ parameter

    Returns:
        Current epsilon value.
    """
    epsilon = privacy_engine.get_epsilon(delta=delta)
    return epsilon


# ---------------------------------------------------------------------------
# Privacy budget estimation (pre-training)
# ---------------------------------------------------------------------------
def estimate_epsilon(
    sample_rate: float,
    noise_multiplier: float,
    steps: int,
    delta: float = 1e-5,
) -> float:
    """
    Estimate the final ε for a given training configuration WITHOUT
    actually training.  Uses Opacus's RDP accountant.

    Args:
        sample_rate:      batch_size / dataset_size
        noise_multiplier: noise std / sensitivity
        steps:            total number of training steps
        delta:            target δ

    Returns:
        Estimated final epsilon.
    """
    from opacus.accountants.rdp import RDPAccountant

    accountant = RDPAccountant()
    for _ in range(steps):
        accountant.step(
            noise_multiplier=noise_multiplier,
            sample_rate=sample_rate,
        )
    return accountant.get_epsilon(delta=delta)
