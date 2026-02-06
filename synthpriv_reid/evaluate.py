"""
evaluate.py — Evaluation for SynthPriv-ReID
=============================================
Two evaluation modes:

1. **Synthetic Image Generation**: Load a trained Generator checkpoint and
   produce a grid of synthetic pedestrian images for visual inspection.

2. **FID (Fréchet Inception Distance)**: Quantitative quality metric.
   NOTE: Computing true FID requires the real dataset AND a pre-trained
   InceptionV3 model. This module provides the full computation pipeline
   using pytorch-fid (if installed) or a structural implementation that
   computes statistics from scratch.

Usage:
  python -m synthpriv_reid.evaluate --checkpoint output/checkpoint_final.pt --num-images 64
  python -m synthpriv_reid.evaluate --checkpoint output/checkpoint_final.pt --compute-fid --real-dir /path/to/real
"""

import argparse
import os
import math

import torch
import numpy as np
from torchvision.utils import save_image

from synthpriv_reid.models import Generator


# ---------------------------------------------------------------------------
# 1. Synthetic image generation
# ---------------------------------------------------------------------------
def generate_synthetic_grid(
    checkpoint_path: str,
    output_path: str = "synthetic_grid.png",
    num_images: int = 64,
    nz: int = 128,
    ngf: int = 64,
    device: str = "cpu",
    seed: int = 42,
) -> str:
    """
    Load a trained Generator and produce a grid of synthetic images.

    Args:
        checkpoint_path: path to the .pt checkpoint file
        output_path:     where to save the image grid
        num_images:      number of images to generate
        nz:              latent dimensionality (must match training)
        ngf:             generator feature maps (must match training)
        device:          cpu or cuda
        seed:            random seed for reproducibility

    Returns:
        Path to the saved grid image.
    """
    torch.manual_seed(seed)

    # Load generator
    generator = Generator(nz=nz, ngf=ngf).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Handle both raw state_dict and full checkpoint formats
    if "generator_state_dict" in ckpt:
        generator.load_state_dict(ckpt["generator_state_dict"])
        epsilon = ckpt.get("epsilon", "unknown")
        epoch = ckpt.get("epoch", "unknown")
        print(f"[Evaluate] Loaded checkpoint from epoch {epoch}, ε = {epsilon}")
    else:
        generator.load_state_dict(ckpt)
        print("[Evaluate] Loaded raw state_dict.")

    generator.eval()

    # Generate images
    with torch.no_grad():
        z = torch.randn(num_images, nz, device=device)
        fake_imgs = generator(z)

    # Denormalize from [-1, 1] to [0, 1] for saving
    fake_imgs = (fake_imgs + 1.0) / 2.0
    fake_imgs = fake_imgs.clamp(0, 1)

    # Save grid
    nrow = int(math.sqrt(num_images))
    if nrow * nrow < num_images:
        nrow = int(math.ceil(math.sqrt(num_images)))
    save_image(fake_imgs, output_path, nrow=nrow, padding=2)
    print(f"[Evaluate] Saved {num_images} synthetic images → {output_path}")

    return output_path


# ---------------------------------------------------------------------------
# 2. Save individual synthetic images (for FID computation)
# ---------------------------------------------------------------------------
def generate_synthetic_dataset(
    checkpoint_path: str,
    output_dir: str = "synthetic_images",
    num_images: int = 1000,
    nz: int = 128,
    ngf: int = 64,
    device: str = "cpu",
    batch_size: int = 64,
    seed: int = 42,
) -> str:
    """
    Generate individual synthetic images and save them to a directory.
    This is needed for FID computation (both pytorch-fid and manual).

    Returns:
        Path to the output directory.
    """
    torch.manual_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    generator = Generator(nz=nz, ngf=ngf).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "generator_state_dict" in ckpt:
        generator.load_state_dict(ckpt["generator_state_dict"])
    else:
        generator.load_state_dict(ckpt)
    generator.eval()

    count = 0
    with torch.no_grad():
        while count < num_images:
            bs = min(batch_size, num_images - count)
            z = torch.randn(bs, nz, device=device)
            fake_imgs = generator(z)
            fake_imgs = (fake_imgs + 1.0) / 2.0
            fake_imgs = fake_imgs.clamp(0, 1)

            for i in range(bs):
                save_image(fake_imgs[i], os.path.join(output_dir, f"synth_{count:05d}.png"))
                count += 1

    print(f"[Evaluate] Generated {count} individual images → {output_dir}/")
    return output_dir


# ---------------------------------------------------------------------------
# 3. FID computation
# ---------------------------------------------------------------------------
def compute_fid_if_available(
    real_dir: str,
    fake_dir: str,
    device: str = "cpu",
    batch_size: int = 50,
) -> float:
    """
    Compute FID between real and synthetic image directories.

    Attempts to use pytorch-fid (pip install pytorch-fid) for a validated
    implementation. If unavailable, computes FID manually using InceptionV3
    feature extraction + Fréchet distance formula.

    The FID formula:
      FID = ||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2(Σ_r Σ_g)^{1/2})

    where (μ_r, Σ_r) and (μ_g, Σ_g) are the mean and covariance of the
    InceptionV3 pool3 features for real and generated images respectively.

    Args:
        real_dir: directory of real images
        fake_dir: directory of generated images
        device:   cpu or cuda
        batch_size: batch size for feature extraction

    Returns:
        FID score (lower is better; 0 = identical distributions)
    """
    # Try pytorch-fid first (most trusted implementation)
    try:
        from pytorch_fid import fid_score
        fid_value = fid_score.calculate_fid_given_paths(
            [real_dir, fake_dir],
            batch_size=batch_size,
            device=torch.device(device),
            dims=2048,
        )
        print(f"[FID] Score (pytorch-fid): {fid_value:.2f}")
        return fid_value
    except ImportError:
        print("[FID] pytorch-fid not installed. Computing FID manually...")
        return _compute_fid_manual(real_dir, fake_dir, device, batch_size)


def _compute_fid_manual(
    real_dir: str,
    fake_dir: str,
    device: str = "cpu",
    batch_size: int = 50,
) -> float:
    """
    Manual FID computation using torchvision's InceptionV3.

    This follows the standard protocol:
    1. Resize images to 299×299 (InceptionV3 input size).
    2. Extract 2048-dim pool3 features.
    3. Compute mean and covariance for real and fake sets.
    4. Compute Fréchet distance.
    """
    from torchvision import models, transforms
    from torch.utils.data import DataLoader
    from torchvision.datasets import ImageFolder
    from scipy import linalg

    # InceptionV3 for feature extraction
    inception = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
    inception.fc = torch.nn.Identity()  # remove final FC — we want pool3 features
    inception = inception.to(device)
    inception.eval()

    preprocess = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def _get_features(img_dir: str) -> np.ndarray:
        """Extract InceptionV3 features from all images in a directory."""
        # ImageFolder requires subdirectory structure; create one if flat
        # Check if images are directly in img_dir
        import glob
        from PIL import Image

        img_paths = sorted(
            glob.glob(os.path.join(img_dir, "*.png"))
            + glob.glob(os.path.join(img_dir, "*.jpg"))
            + glob.glob(os.path.join(img_dir, "*.jpeg"))
        )

        if not img_paths:
            raise FileNotFoundError(f"No images found in {img_dir}")

        all_features = []
        with torch.no_grad():
            for i in range(0, len(img_paths), batch_size):
                batch_paths = img_paths[i:i + batch_size]
                batch_tensors = []
                for p in batch_paths:
                    img = Image.open(p).convert("RGB")
                    batch_tensors.append(preprocess(img))
                batch = torch.stack(batch_tensors).to(device)
                feats = inception(batch)
                if isinstance(feats, tuple):
                    feats = feats[0]  # InceptionV3 returns (logits, aux) in training mode
                all_features.append(feats.cpu().numpy())

        return np.concatenate(all_features, axis=0)

    def _frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
        """Numpy implementation of the Fréchet distance."""
        diff = mu1 - mu2
        covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
        if not np.isfinite(covmean).all():
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        return diff @ diff + np.trace(sigma1 + sigma2 - 2.0 * covmean)

    print(f"[FID] Extracting features from real images: {real_dir}")
    feats_real = _get_features(real_dir)
    print(f"[FID] Extracting features from fake images: {fake_dir}")
    feats_fake = _get_features(fake_dir)

    mu_r, sigma_r = feats_real.mean(axis=0), np.cov(feats_real, rowvar=False)
    mu_f, sigma_f = feats_fake.mean(axis=0), np.cov(feats_fake, rowvar=False)

    fid = _frechet_distance(mu_r, sigma_r, mu_f, sigma_f)
    print(f"[FID] Score (manual): {fid:.2f}")
    return float(fid)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="SynthPriv-ReID Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to Generator checkpoint (.pt)")
    parser.add_argument("--output-dir", type=str, default="output",
                        help="Directory for evaluation outputs")
    parser.add_argument("--num-images", type=int, default=64,
                        help="Number of synthetic images to generate")
    parser.add_argument("--nz", type=int, default=128)
    parser.add_argument("--ngf", type=int, default=64)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)

    # FID options
    parser.add_argument("--compute-fid", action="store_true",
                        help="Compute FID against real images")
    parser.add_argument("--real-dir", type=str, default=None,
                        help="Directory of real images for FID (required if --compute-fid)")
    parser.add_argument("--fid-num-images", type=int, default=1000,
                        help="Number of synthetic images for FID computation")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Generate image grid
    grid_path = os.path.join(args.output_dir, "synthetic_grid.png")
    generate_synthetic_grid(
        checkpoint_path=args.checkpoint,
        output_path=grid_path,
        num_images=args.num_images,
        nz=args.nz,
        ngf=args.ngf,
        device=args.device,
        seed=args.seed,
    )

    # 2. Optionally compute FID
    if args.compute_fid:
        if args.real_dir is None:
            print("[Error] --real-dir is required for FID computation.")
            return

        fake_dir = os.path.join(args.output_dir, "synthetic_for_fid")
        generate_synthetic_dataset(
            checkpoint_path=args.checkpoint,
            output_dir=fake_dir,
            num_images=args.fid_num_images,
            nz=args.nz,
            ngf=args.ngf,
            device=args.device,
            seed=args.seed,
        )

        fid = compute_fid_if_available(
            real_dir=args.real_dir,
            fake_dir=fake_dir,
            device=args.device,
        )
        print(f"\n{'='*40}")
        print(f"  FID Score: {fid:.2f}")
        print(f"{'='*40}")


if __name__ == "__main__":
    main()
