"""
data_pipeline.py — Data Pipeline for SynthPriv-ReID
====================================================
Provides a PyTorch DataLoader for pedestrian images at 64×128 resolution.
Supports two modes:
  1. Real mode: loads Market-1501 dataset from disk.
  2. Dummy mode: generates random tensors for structural testing.

Market-1501 directory structure (expected):
  Market-1501-v15.09.15/
    bounding_box_train/
      0001_c1s1_000151_01.jpg
      ...
"""

import os
from typing import Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


# ---------------------------------------------------------------------------
# Transforms: resize to 64 (W) × 128 (H) and normalise to [-1, 1]
# ---------------------------------------------------------------------------
def get_transforms(img_height: int = 128, img_width: int = 64) -> transforms.Compose:
    """Standard transform pipeline for pedestrian images."""
    return transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),           # → [0, 1]
        transforms.Normalize(            # → [-1, 1]
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        ),
    ])


# ---------------------------------------------------------------------------
# Market-1501 Dataset
# ---------------------------------------------------------------------------
class Market1501Dataset(Dataset):
    """
    Loads images from the Market-1501 bounding_box_train folder.

    Each filename follows the convention: PPPP_CCSS_FFFFFF_NN.jpg
      PPPP  = person identity (0001–1501, plus -1/0000 for junk/distractor)
      CC    = camera id
      SS    = sequence
      FFFFFF = frame number
      NN    = detection index

    We discard identities -1 and 0000 (junk / distractors).
    """

    VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

    def __init__(
        self,
        root: str,
        split: str = "bounding_box_train",
        transform: Optional[transforms.Compose] = None,
    ):
        self.root = os.path.join(root, split)
        self.transform = transform or get_transforms()

        if not os.path.isdir(self.root):
            raise FileNotFoundError(
                f"Market-1501 split directory not found: {self.root}\n"
                f"Please download from: https://zheng-lab.cecs.anu.edu.au/Project/project_reid.html"
            )

        self.samples = []  # list of (path, identity_label)
        self._identity_to_label = {}
        self._load_samples()

    def _load_samples(self):
        """Scan directory and build (path, label) pairs."""
        raw_ids = set()
        entries = []

        for fname in sorted(os.listdir(self.root)):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in self.VALID_EXTENSIONS:
                continue
            # Parse identity from filename
            pid_str = fname.split("_")[0]
            try:
                pid = int(pid_str)
            except ValueError:
                continue
            if pid <= 0:  # skip junk (-1) and distractors (0000)
                continue
            raw_ids.add(pid)
            entries.append((os.path.join(self.root, fname), pid))

        # Map raw identities to contiguous labels 0..N-1
        sorted_ids = sorted(raw_ids)
        self._identity_to_label = {pid: idx for idx, pid in enumerate(sorted_ids)}

        for path, pid in entries:
            self.samples.append((path, self._identity_to_label[pid]))

    @property
    def num_identities(self) -> int:
        return len(self._identity_to_label)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# ---------------------------------------------------------------------------
# Dummy Dataset (for structural / unit testing without real data)
# ---------------------------------------------------------------------------
class DummyPedestrianDataset(Dataset):
    """
    Generates random RGB tensors of shape (3, 128, 64) with random labels.
    Useful for verifying the training pipeline without Market-1501.
    """

    def __init__(
        self,
        num_samples: int = 2048,
        num_identities: int = 50,
        img_height: int = 128,
        img_width: int = 64,
    ):
        self.num_samples = num_samples
        self.num_identities = num_identities
        self.img_height = img_height
        self.img_width = img_width

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        # Random image in [-1, 1]
        img = torch.randn(3, self.img_height, self.img_width)
        label = torch.randint(0, self.num_identities, (1,)).item()
        return img, label


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------
def build_dataloader(
    dataset_root: Optional[str] = None,
    batch_size: int = 64,
    num_workers: int = 2,
    use_dummy: bool = False,
    dummy_samples: int = 2048,
    dummy_identities: int = 50,
) -> Tuple[DataLoader, int]:
    """
    Build a DataLoader for either real Market-1501 or dummy data.

    Returns:
        dataloader: PyTorch DataLoader
        num_identities: number of distinct person identities
    """
    if use_dummy:
        dataset = DummyPedestrianDataset(
            num_samples=dummy_samples,
            num_identities=dummy_identities,
        )
        num_identities = dummy_identities
    else:
        if dataset_root is None:
            raise ValueError(
                "dataset_root must be provided when use_dummy=False. "
                "Point it to the Market-1501-v15.09.15 directory."
            )
        dataset = Market1501Dataset(root=dataset_root)
        num_identities = dataset.num_identities

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,   # required for consistent batch sizes with Opacus
        pin_memory=True,
    )
    return dataloader, num_identities
