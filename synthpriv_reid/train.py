"""
train.py — Adversarial Training Loop with Differential Privacy
===============================================================
Trains the DP-GAN:
  1. Discriminator step (with DP-SGD via Opacus)
  2. Generator step (standard backprop — no DP needed)
  3. Privacy budget tracking after every epoch

Usage:
  python -m synthpriv_reid.train --use-dummy --epochs 20
  python -m synthpriv_reid.train --dataset-root /path/to/Market-1501-v15.09.15 --epochs 50
"""

import argparse
import json
import os
import time
from typing import Dict, List

import torch
import torch.nn as nn
from torchvision.utils import save_image

from synthpriv_reid.data_pipeline import build_dataloader
from synthpriv_reid.models import Generator, Discriminator
from synthpriv_reid.privacy_engine import (
    make_discriminator_private,
    get_current_epsilon,
    estimate_epsilon,
)


# ---------------------------------------------------------------------------
# Training configuration dataclass
# ---------------------------------------------------------------------------
class TrainConfig:
    """All hyperparameters in one place."""

    # Data
    dataset_root: str = None
    use_dummy: bool = True
    batch_size: int = 64
    num_workers: int = 2

    # Model
    nz: int = 128           # latent vector dimensionality
    ngf: int = 64           # generator feature maps
    ndf: int = 64           # discriminator feature maps

    # Training
    epochs: int = 50
    lr_g: float = 2e-4      # generator learning rate
    lr_d: float = 2e-4      # discriminator learning rate
    beta1: float = 0.5      # Adam β1
    beta2: float = 0.999    # Adam β2

    # Differential privacy
    target_epsilon: float = 8.0
    target_delta: float = 1e-5
    max_grad_norm: float = 1.0
    noise_multiplier: float = 1.1

    # Output
    output_dir: str = "output"
    save_every: int = 5     # save checkpoints every N epochs
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Training engine
# ---------------------------------------------------------------------------
class Trainer:
    """Orchestrates DP-GAN training."""

    def __init__(self, config: TrainConfig):
        self.cfg = config
        self.device = torch.device(config.device)
        os.makedirs(config.output_dir, exist_ok=True)

        # ---- Data ----
        self.dataloader, self.num_ids = build_dataloader(
            dataset_root=config.dataset_root,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            use_dummy=config.use_dummy,
        )
        print(f"[Trainer] Dataset: {'dummy' if config.use_dummy else config.dataset_root}")
        print(f"[Trainer] Identities: {self.num_ids}  |  Batches/epoch: {len(self.dataloader)}")

        # ---- Models ----
        self.generator = Generator(nz=config.nz, ngf=config.ngf).to(self.device)
        self.discriminator = Discriminator(ndf=config.ndf).to(self.device)

        # ---- Optimisers ----
        self.opt_g = torch.optim.Adam(
            self.generator.parameters(),
            lr=config.lr_g,
            betas=(config.beta1, config.beta2),
        )
        self.opt_d = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=config.lr_d,
            betas=(config.beta1, config.beta2),
        )

        # ---- Privacy ----
        (
            self.discriminator,
            self.opt_d,
            self.dataloader,
            self.privacy_engine,
        ) = make_discriminator_private(
            discriminator=self.discriminator,
            optimizer=self.opt_d,
            dataloader=self.dataloader,
            target_epsilon=config.target_epsilon,
            target_delta=config.target_delta,
            max_grad_norm=config.max_grad_norm,
            noise_multiplier=config.noise_multiplier,
            epochs=config.epochs,
        )

        # ---- Loss ----
        self.criterion = nn.BCEWithLogitsLoss()

        # ---- Tracking ----
        self.history: Dict[str, List[float]] = {
            "loss_d": [],
            "loss_g": [],
            "epsilon": [],
        }

        # Pre-training epsilon estimate
        dataset_size = len(self.dataloader.dataset)
        sample_rate = config.batch_size / dataset_size
        total_steps = config.epochs * len(self.dataloader)
        est_eps = estimate_epsilon(
            sample_rate=sample_rate,
            noise_multiplier=config.noise_multiplier,
            steps=total_steps,
            delta=config.target_delta,
        )
        print(f"[Trainer] Estimated final ε ≈ {est_eps:.2f}  (target < {config.target_epsilon})")

    # ---- Single training step: Discriminator ----
    def _step_discriminator(self, real_imgs: torch.Tensor) -> float:
        """
        One DP-SGD step for the discriminator.

        Follows the official Opacus DCGAN pattern:
          zero_grad → forward(real) → forward(fake) → combined backward → step
        This ensures Opacus's per-sample gradient hooks fire correctly.
        """
        batch_size = real_imgs.size(0)
        self.opt_d.zero_grad(set_to_none=True)

        # Real images → should be classified as 1
        label_real = torch.ones(batch_size, 1, device=self.device)
        output_real = self.discriminator(real_imgs)
        loss_real = self.criterion(output_real, label_real)

        # Fake images → should be classified as 0
        noise = torch.randn(batch_size, self.cfg.nz, device=self.device)
        with torch.no_grad():
            fake_imgs = self.generator(noise)
        label_fake = torch.zeros(batch_size, 1, device=self.device)
        output_fake = self.discriminator(fake_imgs.detach())
        loss_fake = self.criterion(output_fake, label_fake)

        # Combined loss — single backward (Opacus requirement)
        loss_d = loss_real + loss_fake
        loss_d.backward()
        self.opt_d.step()

        return loss_d.item()

    # ---- Single training step: Generator ----
    def _step_generator(self, batch_size: int) -> float:
        """Standard (non-private) step for the generator."""
        self.opt_g.zero_grad(set_to_none=True)

        noise = torch.randn(batch_size, self.cfg.nz, device=self.device)
        fake_imgs = self.generator(noise)
        label_real = torch.ones(batch_size, 1, device=self.device)

        # Generator wants discriminator to classify fakes as real
        output = self.discriminator(fake_imgs)
        loss_g = self.criterion(output, label_real)
        loss_g.backward()
        self.opt_g.step()

        return loss_g.item()

    # ---- Full training loop ----
    def train(self):
        """Run the full training loop with privacy budget tracking."""
        print(f"\n{'='*60}")
        print(f"  SynthPriv-ReID  —  DP-GAN Training")
        print(f"  Device: {self.device}  |  Epochs: {self.cfg.epochs}")
        print(f"{'='*60}\n")

        fixed_noise = torch.randn(16, self.cfg.nz, device=self.device)

        for epoch in range(1, self.cfg.epochs + 1):
            t0 = time.time()
            epoch_loss_d = 0.0
            epoch_loss_g = 0.0
            num_batches = 0

            self.generator.train()
            self.discriminator.train()

            for real_imgs, _labels in self.dataloader:
                real_imgs = real_imgs.to(self.device)
                bs = real_imgs.size(0)

                # --- Discriminator (with DP) ---
                loss_d = self._step_discriminator(real_imgs)

                # --- Generator (standard) ---
                loss_g = self._step_generator(bs)

                epoch_loss_d += loss_d
                epoch_loss_g += loss_g
                num_batches += 1

            # Average losses
            avg_loss_d = epoch_loss_d / max(num_batches, 1)
            avg_loss_g = epoch_loss_g / max(num_batches, 1)

            # Query privacy budget
            epsilon = get_current_epsilon(self.privacy_engine, delta=self.cfg.target_delta)

            self.history["loss_d"].append(avg_loss_d)
            self.history["loss_g"].append(avg_loss_g)
            self.history["epsilon"].append(epsilon)

            elapsed = time.time() - t0
            print(
                f"Epoch [{epoch:3d}/{self.cfg.epochs}]  "
                f"Loss_D: {avg_loss_d:.4f}  Loss_G: {avg_loss_g:.4f}  "
                f"ε: {epsilon:.2f}  (δ={self.cfg.target_delta})  "
                f"Time: {elapsed:.1f}s"
            )

            # Early stopping if privacy budget exceeded
            if epsilon > self.cfg.target_epsilon:
                print(f"\n[!] Privacy budget exhausted (ε={epsilon:.2f} > {self.cfg.target_epsilon}). Stopping.")
                break

            # Periodic checkpoint + sample generation
            if epoch % self.cfg.save_every == 0:
                self._save_checkpoint(epoch, epsilon)
                self._save_samples(epoch, fixed_noise)

        # Final checkpoint + samples + training curves
        self._save_checkpoint(epoch, epsilon, final=True)
        self._save_samples(epoch, fixed_noise)
        self._save_training_curves()
        print(f"\nTraining complete. Final ε = {epsilon:.2f}")
        return self.history

    # ---- Checkpointing ----
    def _save_checkpoint(self, epoch: int, epsilon: float, final: bool = False):
        """Save model weights and training state."""
        tag = "final" if final else f"epoch{epoch:03d}"
        path = os.path.join(self.cfg.output_dir, f"checkpoint_{tag}.pt")
        torch.save({
            "epoch": epoch,
            "epsilon": epsilon,
            "generator_state_dict": self.generator.state_dict(),
            "discriminator_state_dict": self.discriminator.state_dict(),
            "optimizer_g_state_dict": self.opt_g.state_dict(),
            "optimizer_d_state_dict": self.opt_d.state_dict(),
            "config": vars(self.cfg),
            "history": self.history,
        }, path)
        print(f"  → Checkpoint saved: {path}")

    # ---- Sample generation ----
    def _save_samples(self, epoch: int, fixed_noise: torch.Tensor):
        """Generate and save a grid of samples for visual tracking."""
        self.generator.eval()
        with torch.no_grad():
            fake = self.generator(fixed_noise)
        fake = (fake + 1.0) / 2.0  # [-1,1] → [0,1]
        fake = fake.clamp(0, 1)
        path = os.path.join(self.cfg.output_dir, f"samples_epoch{epoch:03d}.png")
        save_image(fake, path, nrow=4, padding=2)
        print(f"  → Samples saved: {path}")

    # ---- Training curve export ----
    def _save_training_curves(self):
        """Save training history as JSON for external plotting."""
        path = os.path.join(self.cfg.output_dir, "training_history.json")
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"  → Training curves saved: {path}")

        # Also attempt matplotlib plot if available
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            epochs = range(1, len(self.history["loss_d"]) + 1)

            axes[0].plot(epochs, self.history["loss_d"], label="D loss")
            axes[0].plot(epochs, self.history["loss_g"], label="G loss")
            axes[0].set_xlabel("Epoch")
            axes[0].set_ylabel("Loss")
            axes[0].set_title("Adversarial Losses")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            axes[1].plot(epochs, self.history["epsilon"], color="red")
            axes[1].axhline(y=self.cfg.target_epsilon, color="gray",
                            linestyle="--", label=f"Target ε={self.cfg.target_epsilon}")
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("ε")
            axes[1].set_title("Privacy Budget Spent")
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

            axes[2].plot(epochs, self.history["loss_d"], label="D loss", alpha=0.7)
            axes[2].set_xlabel("Epoch")
            axes[2].set_ylabel("Loss_D")
            axes[2].set_title("Discriminator Convergence")
            axes[2].grid(True, alpha=0.3)

            plt.tight_layout()
            plot_path = os.path.join(self.cfg.output_dir, "training_curves.png")
            fig.savefig(plot_path, dpi=150)
            plt.close(fig)
            print(f"  → Training plot saved: {plot_path}")
        except ImportError:
            print("  (matplotlib not installed — skipping plot generation)")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="SynthPriv-ReID DP-GAN Training")
    parser.add_argument("--dataset-root", type=str, default=None,
                        help="Path to Market-1501-v15.09.15 directory")
    parser.add_argument("--use-dummy", action="store_true",
                        help="Use synthetic dummy data for testing")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--nz", type=int, default=128, help="Latent vector size")
    parser.add_argument("--lr-g", type=float, default=2e-4)
    parser.add_argument("--lr-d", type=float, default=2e-4)
    parser.add_argument("--noise-multiplier", type=float, default=1.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--target-epsilon", type=float, default=8.0)
    parser.add_argument("--target-delta", type=float, default=1e-5)
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = TrainConfig()
    cfg.dataset_root = args.dataset_root
    cfg.use_dummy = args.use_dummy or (args.dataset_root is None)
    cfg.batch_size = args.batch_size
    cfg.epochs = args.epochs
    cfg.nz = args.nz
    cfg.lr_g = args.lr_g
    cfg.lr_d = args.lr_d
    cfg.noise_multiplier = args.noise_multiplier
    cfg.max_grad_norm = args.max_grad_norm
    cfg.target_epsilon = args.target_epsilon
    cfg.target_delta = args.target_delta
    cfg.output_dir = args.output_dir
    cfg.device = args.device
    return cfg


if __name__ == "__main__":
    config = parse_args()
    trainer = Trainer(config)
    trainer.train()
