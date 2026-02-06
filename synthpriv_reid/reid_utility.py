"""
reid_utility.py — Downstream Re-ID Utility Validation
=======================================================
Measures the practical value of DP-synthetic data by training a standard
Person Re-ID model (ResNet-50 backbone) on synthetic images and testing
it on real Market-1501 queries.

This module answers the fundamental question: "How much utility do we
retain after applying differential privacy to the generative process?"

Protocol:
  1. Generate a labelled synthetic dataset using the trained DP-GAN.
  2. Train a ResNet-50 Re-ID classifier on the synthetic identities.
  3. Extract feature embeddings for real query/gallery images.
  4. Evaluate Rank-1 accuracy and mAP (mean Average Precision).

Usage:
  python -m synthpriv_reid.reid_utility \
    --checkpoint output/checkpoint_final.pt \
    --dataset-root /path/to/Market-1501-v15.09.15 \
    --num-synthetic 5000 \
    --epochs 20
"""

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from synthpriv_reid.models import Generator


# ---------------------------------------------------------------------------
# Synthetic labelled dataset from the trained Generator
# ---------------------------------------------------------------------------
class SyntheticReIDDataset(Dataset):
    """
    Generates labelled synthetic images on-the-fly from the Generator.

    Each "identity" is defined by a fixed latent anchor; variations are
    produced by adding small Gaussian perturbations around the anchor.
    This simulates multiple views of the same synthetic person.
    """

    def __init__(
        self,
        generator: nn.Module,
        num_identities: int = 100,
        images_per_id: int = 50,
        nz: int = 128,
        perturbation_std: float = 0.3,
        device: str = "cpu",
        seed: int = 42,
    ):
        self.generator = generator
        self.nz = nz
        self.device = device
        self.num_identities = num_identities
        self.images_per_id = images_per_id
        self.total = num_identities * images_per_id

        # Create fixed identity anchors in latent space
        rng = torch.Generator().manual_seed(seed)
        self.anchors = torch.randn(num_identities, nz, generator=rng)
        self.perturbation_std = perturbation_std

    def __len__(self) -> int:
        return self.total

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        identity = idx // self.images_per_id
        anchor = self.anchors[identity]

        # Perturb the anchor to create an appearance variation
        noise = anchor + self.perturbation_std * torch.randn(self.nz)
        noise = noise.unsqueeze(0).to(self.device)

        with torch.no_grad():
            img = self.generator(noise).squeeze(0).cpu()

        # Resize to 224x224 for ResNet-50 and renormalize
        # Generator outputs [-1,1]; ResNet expects ImageNet normalization
        img = (img + 1.0) / 2.0  # → [0, 1]
        img = img.clamp(0, 1)

        return img, identity


# ---------------------------------------------------------------------------
# ResNet-50 Re-ID backbone
# ---------------------------------------------------------------------------
class ReIDModel(nn.Module):
    """
    ResNet-50 backbone adapted for Person Re-ID.

    The final FC layer is replaced with:
      1. A 512-d embedding layer (for feature extraction at test time)
      2. A classification head (for training with cross-entropy)
    """

    def __init__(self, num_classes: int, embedding_dim: int = 512):
        super().__init__()
        backbone = models.resnet50(weights=None)
        # Remove the original FC
        self.features = nn.Sequential(*list(backbone.children())[:-1])  # → (B, 2048, 1, 1)
        self.embedding = nn.Linear(2048, embedding_dim)
        self.bn = nn.BatchNorm1d(embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        """
        Args:
            x: image tensor (B, 3, H, W)
            return_embedding: if True, return L2-normalized embedding only
        """
        feat = self.features(x).flatten(1)       # (B, 2048)
        emb = self.bn(self.embedding(feat))       # (B, 512)

        if return_embedding:
            return nn.functional.normalize(emb, p=2, dim=1)

        logits = self.classifier(emb)
        return logits


# ---------------------------------------------------------------------------
# Transform for ResNet-50 input
# ---------------------------------------------------------------------------
REID_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Transform for synthetic images (already tensors in [0,1])
SYNTH_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ---------------------------------------------------------------------------
# Training the Re-ID model on synthetic data
# ---------------------------------------------------------------------------
def train_reid_on_synthetic(
    generator_checkpoint: str,
    num_identities: int = 100,
    images_per_id: int = 50,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 3e-4,
    nz: int = 128,
    ngf: int = 64,
    device: str = "cpu",
    output_dir: str = "output",
) -> nn.Module:
    """
    Train a ResNet-50 Re-ID model on synthetic data from the DP-GAN.

    Returns:
        Trained ReIDModel.
    """
    print(f"\n{'='*60}")
    print(f"  Re-ID Utility Validation — Training Phase")
    print(f"  Synthetic identities: {num_identities}")
    print(f"  Images per identity:  {images_per_id}")
    print(f"  Total synthetic images: {num_identities * images_per_id}")
    print(f"{'='*60}\n")

    # Load generator
    gen = Generator(nz=nz, ngf=ngf).to(device)
    ckpt = torch.load(generator_checkpoint, map_location=device, weights_only=False)
    if "generator_state_dict" in ckpt:
        gen.load_state_dict(ckpt["generator_state_dict"])
        epsilon = ckpt.get("epsilon", "?")
        print(f"[ReID] Generator loaded (ε = {epsilon})")
    else:
        gen.load_state_dict(ckpt)
    gen.eval()

    # Build synthetic dataset
    synth_dataset = SyntheticReIDDataset(
        generator=gen,
        num_identities=num_identities,
        images_per_id=images_per_id,
        nz=nz,
        device=device,
    )
    synth_loader = DataLoader(
        synth_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, drop_last=True,
    )

    # Build Re-ID model
    reid_model = ReIDModel(num_classes=num_identities).to(device)
    optimizer = torch.optim.Adam(reid_model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(1, epochs + 1):
        reid_model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for imgs, labels in synth_loader:
            # Apply ImageNet normalization to synthetic images
            imgs = SYNTH_TRANSFORM(imgs).to(device)
            labels = labels.to(device)

            logits = reid_model(imgs)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            _, predicted = logits.max(1)
            correct += predicted.eq(labels).sum().item()
            total += imgs.size(0)

        acc = 100.0 * correct / total
        avg_loss = total_loss / total
        print(f"  Epoch [{epoch:2d}/{epochs}]  Loss: {avg_loss:.4f}  Acc: {acc:.1f}%")

    # Save Re-ID model
    reid_path = os.path.join(output_dir, "reid_model.pt")
    torch.save(reid_model.state_dict(), reid_path)
    print(f"[ReID] Model saved → {reid_path}")

    return reid_model


# ---------------------------------------------------------------------------
# Evaluation metrics: Rank-1 accuracy and mAP
# ---------------------------------------------------------------------------
def extract_features(
    model: ReIDModel,
    dataloader: DataLoader,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract L2-normalized embeddings for all images in the dataloader."""
    model.eval()
    all_features = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            emb = model(imgs, return_embedding=True)
            all_features.append(emb.cpu().numpy())
            all_labels.append(labels.numpy())

    return np.concatenate(all_features), np.concatenate(all_labels)


def compute_cmc_map(
    query_features: np.ndarray,
    query_labels: np.ndarray,
    gallery_features: np.ndarray,
    gallery_labels: np.ndarray,
    top_k: int = 10,
) -> Dict[str, float]:
    """
    Compute Cumulative Match Characteristic (CMC) and mean Average Precision.

    Args:
        query_features:   (N_q, D) L2-normalized feature vectors
        query_labels:     (N_q,) identity labels
        gallery_features: (N_g, D) L2-normalized feature vectors
        gallery_labels:   (N_g,) identity labels
        top_k:            compute Rank-1 through Rank-k

    Returns:
        Dict with 'rank1', 'rank5', 'rank10', 'mAP'.
    """
    # Cosine similarity (features are L2-normalized)
    dist_matrix = query_features @ gallery_features.T  # (N_q, N_g)

    cmc = np.zeros(top_k)
    ap_sum = 0.0
    num_valid = 0

    for i in range(len(query_labels)):
        q_label = query_labels[i]
        scores = dist_matrix[i]

        # Sort gallery by descending similarity
        ranked_indices = np.argsort(-scores)
        ranked_labels = gallery_labels[ranked_indices]

        # CMC: does the correct identity appear in top-k?
        matches = (ranked_labels == q_label).astype(np.float32)
        if matches.sum() == 0:
            continue  # skip queries with no gallery match

        num_valid += 1

        # Rank-k accuracy
        for k in range(top_k):
            if matches[:k + 1].sum() > 0:
                cmc[k] += 1

        # Average Precision for this query
        num_relevant = matches.sum()
        cum_matches = np.cumsum(matches)
        precision_at_k = cum_matches * matches / (np.arange(len(matches)) + 1)
        ap = precision_at_k.sum() / num_relevant
        ap_sum += ap

    if num_valid == 0:
        return {"rank1": 0.0, "rank5": 0.0, "rank10": 0.0, "mAP": 0.0}

    cmc /= num_valid
    mAP = ap_sum / num_valid

    results = {
        "rank1": float(cmc[0]) * 100,
        "rank5": float(cmc[min(4, top_k - 1)]) * 100,
        "rank10": float(cmc[min(9, top_k - 1)]) * 100,
        "mAP": float(mAP) * 100,
    }
    return results


# ---------------------------------------------------------------------------
# Full evaluation pipeline
# ---------------------------------------------------------------------------
def evaluate_reid_utility(
    reid_model: ReIDModel,
    dataset_root: str,
    device: str = "cpu",
    batch_size: int = 64,
) -> Dict[str, float]:
    """
    Evaluate the Re-ID model trained on synthetic data against real
    Market-1501 query/gallery splits.

    Args:
        reid_model:   trained ReIDModel
        dataset_root: path to Market-1501-v15.09.15
        device:       cpu or cuda
        batch_size:   batch size for feature extraction

    Returns:
        Dict with Rank-1, Rank-5, Rank-10, and mAP.
    """
    from synthpriv_reid.data_pipeline import Market1501Dataset

    print(f"\n{'='*60}")
    print(f"  Re-ID Utility Validation — Evaluation Phase")
    print(f"{'='*60}\n")

    transform = REID_TRANSFORM

    # Load query and gallery splits
    query_dataset = Market1501Dataset(
        root=dataset_root, split="query", transform=transform,
    )
    gallery_dataset = Market1501Dataset(
        root=dataset_root, split="bounding_box_test", transform=transform,
    )

    query_loader = DataLoader(query_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    gallery_loader = DataLoader(gallery_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    print(f"  Query images:   {len(query_dataset)}")
    print(f"  Gallery images: {len(gallery_dataset)}")

    # Extract features
    print("  Extracting query features...")
    q_feats, q_labels = extract_features(reid_model, query_loader, device)
    print("  Extracting gallery features...")
    g_feats, g_labels = extract_features(reid_model, gallery_loader, device)

    # Compute metrics
    metrics = compute_cmc_map(q_feats, q_labels, g_feats, g_labels)

    print(f"\n  Results (Synthetic → Real transfer):")
    print(f"  {'─'*40}")
    print(f"  Rank-1:  {metrics['rank1']:.2f}%")
    print(f"  Rank-5:  {metrics['rank5']:.2f}%")
    print(f"  Rank-10: {metrics['rank10']:.2f}%")
    print(f"  mAP:     {metrics['mAP']:.2f}%")
    print(f"  {'─'*40}\n")

    return metrics


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="SynthPriv-ReID: Downstream Re-ID Utility Evaluation"
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to DP-GAN Generator checkpoint (.pt)")
    parser.add_argument("--dataset-root", type=str, default=None,
                        help="Path to Market-1501-v15.09.15 (for real evaluation)")
    parser.add_argument("--num-identities", type=int, default=100,
                        help="Number of synthetic identities to generate")
    parser.add_argument("--images-per-id", type=int, default=50,
                        help="Number of images per synthetic identity")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Training epochs for the Re-ID model")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--nz", type=int, default=128)
    parser.add_argument("--ngf", type=int, default=64)
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Phase 1: Train Re-ID model on synthetic data
    reid_model = train_reid_on_synthetic(
        generator_checkpoint=args.checkpoint,
        num_identities=args.num_identities,
        images_per_id=args.images_per_id,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        nz=args.nz,
        ngf=args.ngf,
        device=args.device,
        output_dir=args.output_dir,
    )

    # Phase 2: Evaluate on real data (if available)
    if args.dataset_root is not None:
        metrics = evaluate_reid_utility(
            reid_model=reid_model,
            dataset_root=args.dataset_root,
            device=args.device,
            batch_size=args.batch_size,
        )

        # Save metrics
        import json
        metrics_path = os.path.join(args.output_dir, "reid_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"[ReID] Metrics saved → {metrics_path}")
    else:
        print("\n[ReID] No --dataset-root provided. Skipping real-data evaluation.")
        print("       To measure privacy-utility trade-off, provide Market-1501 path.")


if __name__ == "__main__":
    main()
