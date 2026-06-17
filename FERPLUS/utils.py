import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2,
               device: torch.device = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def cutmix_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0,
                device: torch.device = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=device)

    h, w = x.size(-2), x.size(-1)
    cut_ratio = np.sqrt(1. - lam)
    cut_w = int(w * cut_ratio)
    cut_h = int(h * cut_ratio)

    cx = np.random.randint(w)
    cy = np.random.randint(h)

    bbx1 = np.clip(cx - cut_w // 2, 0, w)
    bby1 = np.clip(cy - cut_h // 2, 0, h)
    bbx2 = np.clip(cx + cut_w // 2, 0, w)
    bby2 = np.clip(cy + cut_h // 2, 0, h)

    mixed_x = x.clone()
    mixed_x[..., bby1:bby2, bbx1:bbx2] = x[index, ..., bby1:bby2, bbx1:bbx2]

    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion: nn.Module, pred: torch.Tensor,
                    y_a: torch.Tensor, y_b: torch.Tensor, lam: float, epoch: int = 0) -> torch.Tensor:
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
