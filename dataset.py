import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from typing import Optional


def worker_init_fn(worker_id: int, seed: int = 42) -> None:
    worker_seed = seed + worker_id
    torch.manual_seed(worker_seed)
    import random
    import numpy as np
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def load_data(path: str, batch_size: int, num_workers: int,
              mode: str = 'train', seed: int = 42) -> DataLoader:
    if mode == "train":
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.RandomResizedCrop(48, scale=(0.88, 1.0), ratio=(0.95, 1.05)),
            transforms.RandomHorizontalFlip(p=0.5),

            transforms.RandomApply([transforms.RandomRotation(degrees=12)
            ], p=0.4),

            transforms.RandomApply([
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=8)
            ], p=0.35),

            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.2, contrast=0.2)
            ], p=0.3),

            transforms.ToTensor(),

            transforms.Normalize(mean=(0.5,), std=(0.5,)),

            transforms.RandomErasing(p=0.2,
                scale=(0.02, 0.15),
                ratio=(0.3, 3.3),
                value=0.0
            ),
        ])
        shuffle = True
    else:
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(48),
            transforms.CenterCrop(48),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,)),
        ])
        shuffle = False

    dataset = ImageFolder(path, transform=transform)

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=4 if num_workers > 0 else None,
        worker_init_fn=worker_init_fn if num_workers > 0 else None,
        generator=generator if shuffle else None,
        drop_last=shuffle,
    )
