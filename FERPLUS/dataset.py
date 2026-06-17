import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from typing import Optional
import pandas as pd

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class FERPlusDataset(Dataset):
    def __init__(self, csv_path, usage='Training', transform=None):
        df = pd.read_csv(csv_path)

        self.data = df[df['Usage'] == usage].reset_index(drop=True)
        self.transform = transform

        self.classes = [
            'neutral', 'happiness', 'surprise', 'sadness', 
            'anger', 'disgust', 'fear', 'contempt', 'unknown', 'NF'
        ]
        self.classes_8 = self.classes[:8]

        label_df = self.data[self.classes]
        mask = label_df[self.classes_8].sum(axis=1) > label_df[['unknown', 'NF']].sum(axis=1)
        self.data = self.data[mask].reset_index(drop=True)

        class_to_idx = {cls: i for i, cls in enumerate(self.classes_8)}
        self.targets = self.data[self.classes_8].idxmax(axis=1).map(class_to_idx).values.tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        pixels = np.fromstring(row['pixel'], sep=' ', dtype=np.uint8).reshape(48, 48)
        img = Image.fromarray(pixels)

        labels = row[self.classes].values.astype(np.float32)
        
        valid_labels = labels[:8]
        
        sum_val = np.sum(valid_labels)
        if sum_val > 0:
            valid_labels = valid_labels / sum_val
        else:
            valid_labels = np.ones(8) / 8.0
            
        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(valid_labels, dtype=torch.float32)

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
        usage = 'Training'
        shuffle = True
    elif mode == "val":
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(48),
            transforms.CenterCrop(48),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,)),
        ])
        usage = 'PublicTest'
        shuffle = False
    else:  
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(48),
            transforms.CenterCrop(48),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,)),
        ])
        usage = 'PrivateTest'
        shuffle = False

    dataset = FERPlusDataset(csv_path=path, usage=usage, transform=transform)

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=4 if num_workers > 0 else None,
        worker_init_fn=worker_init_fn if num_workers > 0 else None,
        generator=generator if shuffle else None,
        drop_last=shuffle,
    )


