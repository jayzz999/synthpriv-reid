"""
models.py — Generator & Discriminator for SynthPriv-ReID
=========================================================
Architecture notes:
  • Output image size: 64 (W) × 128 (H) × 3 (RGB), values in [-1, 1].
  • The Discriminator uses GroupNorm instead of BatchNorm because Opacus
    requires per-sample gradient computation, which is incompatible with
    BatchNorm (batch statistics break the independence assumption of DP).
  • The Generator is NOT wrapped by Opacus — it never sees real data
    directly — so it may use BatchNorm freely. We still use GroupNorm
    for consistency and to avoid issues if the architecture is reused.

Shape walkthrough (Generator):
  z ∈ ℝ^{nz}  →  FC  →  (ngf*8, 8, 4)
    → ConvT → (ngf*4, 16, 8)
    → ConvT → (ngf*2, 32, 16)
    → ConvT → (ngf,   64, 32)
    → ConvT → (3,    128, 64)   ← output

Shape walkthrough (Discriminator):
  x ∈ ℝ^{3×128×64}
    → Conv → (ndf,   64, 32)
    → Conv → (ndf*2, 32, 16)
    → Conv → (ndf*4, 16, 8)
    → Conv → (ndf*8, 8,  4)
    → FC   → 1                  ← real/fake logit
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Weight initialisation (standard DCGAN practice)
# ---------------------------------------------------------------------------
def weights_init(m: nn.Module):
    """Apply DCGAN-style weight initialisation."""
    classname = m.__class__.__name__
    if "Conv" in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "GroupNorm" in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif "Linear" in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------
class Generator(nn.Module):
    """
    Transposed-convolution generator.

    Input:  noise vector z of shape (batch, nz, 1, 1)
    Output: RGB image of shape (batch, 3, 128, 64)
    """

    def __init__(self, nz: int = 128, ngf: int = 64, nc: int = 3):
        """
        Args:
            nz:  dimensionality of the latent noise vector
            ngf: base number of generator feature maps
            nc:  number of output channels (3 for RGB)
        """
        super().__init__()
        self.nz = nz
        self.ngf = ngf

        self.main = nn.Sequential(
            # ------ Block 1: project & reshape ------
            # Input: (nz, 1, 1) → (ngf*8, 8, 4)
            nn.ConvTranspose2d(nz, ngf * 8, kernel_size=(8, 4), stride=1, padding=0, bias=False),
            nn.GroupNorm(8, ngf * 8),
            nn.ReLU(True),

            # ------ Block 2 ------
            # (ngf*8, 8, 4) → (ngf*4, 16, 8)
            nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, ngf * 4),
            nn.ReLU(True),

            # ------ Block 3 ------
            # (ngf*4, 16, 8) → (ngf*2, 32, 16)
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, ngf * 2),
            nn.ReLU(True),

            # ------ Block 4 ------
            # (ngf*2, 32, 16) → (ngf, 64, 32)
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, ngf),
            nn.ReLU(True),

            # ------ Block 5: to image ------
            # (ngf, 64, 32) → (nc, 128, 64)
            nn.ConvTranspose2d(ngf, nc, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),  # output in [-1, 1]
        )

        self.apply(weights_init)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: latent vector, shape (batch, nz) or (batch, nz, 1, 1)
        Returns:
            Generated image, shape (batch, 3, 128, 64)
        """
        if z.dim() == 2:
            z = z.unsqueeze(-1).unsqueeze(-1)  # (B, nz) → (B, nz, 1, 1)
        return self.main(z)


# ---------------------------------------------------------------------------
# Discriminator
# ---------------------------------------------------------------------------
class Discriminator(nn.Module):
    """
    CNN discriminator for real-vs-fake classification.

    Input:  RGB image of shape (batch, 3, 128, 64)
    Output: single logit per image (no sigmoid — use BCEWithLogitsLoss)

    Uses GroupNorm throughout for Opacus compatibility.
    Uses LeakyReLU (slope 0.2) as is standard for discriminators.
    """

    def __init__(self, ndf: int = 64, nc: int = 3):
        """
        Args:
            ndf: base number of discriminator feature maps
            nc:  number of input channels (3 for RGB)
        """
        super().__init__()
        self.ndf = ndf

        self.features = nn.Sequential(
            # ------ Block 1 (no norm on first layer — standard practice) ------
            # (nc, 128, 64) → (ndf, 64, 32)
            nn.Conv2d(nc, ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=False),

            # ------ Block 2 ------
            # (ndf, 64, 32) → (ndf*2, 32, 16)
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, ndf * 2),
            nn.LeakyReLU(0.2, inplace=False),

            # ------ Block 3 ------
            # (ndf*2, 32, 16) → (ndf*4, 16, 8)
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, ndf * 4),
            nn.LeakyReLU(0.2, inplace=False),

            # ------ Block 4 ------
            # (ndf*4, 16, 8) → (ndf*8, 8, 4)
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, ndf * 8),
            nn.LeakyReLU(0.2, inplace=False),
        )

        # Final classification head: (ndf*8, 8, 4) → 1
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(ndf * 8 * 8 * 4, 1),
        )

        self.apply(weights_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: image tensor, shape (batch, 3, 128, 64)
        Returns:
            logits, shape (batch, 1)
        """
        features = self.features(x)
        return self.classifier(features)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    nz = 128
    batch = 4

    G = Generator(nz=nz, ngf=64)
    D = Discriminator(ndf=64)

    z = torch.randn(batch, nz)
    fake = G(z)
    print(f"Generator output shape: {fake.shape}")  # expect (4, 3, 128, 64)

    logit = D(fake)
    print(f"Discriminator output shape: {logit.shape}")  # expect (4, 1)

    # Count parameters
    g_params = sum(p.numel() for p in G.parameters())
    d_params = sum(p.numel() for p in D.parameters())
    print(f"Generator params:     {g_params:,}")
    print(f"Discriminator params: {d_params:,}")
