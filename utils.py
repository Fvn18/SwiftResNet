import os
import random
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SCELoss(nn.Module):

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.2,
        num_classes: int = 7,
        label_smoothing: float = 0.1,
        eps: float = 1e-7,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        self.eps = eps

    def forward(
        self,
        pred: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        ce = F.cross_entropy(pred, labels,label_smoothing=self.label_smoothing)

        smooth_value = self.label_smoothing / self.num_classes

        one_hot = torch.full((pred.size(0), self.num_classes), smooth_value,
            device=pred.device, dtype=pred.dtype,
        )

        one_hot.scatter_(1, labels.unsqueeze(1),
            1.0 - self.label_smoothing + smooth_value,
        )

        one_hot = one_hot.clamp(
            min=self.eps,
            max=1.0,
        )

        pred_prob = F.softmax(pred, dim=1).clamp(
            min=self.eps,
            max=1.0 - self.eps,
        )

        rce = -torch.sum(
            pred_prob * torch.log(one_hot),
            dim=1,
        ).mean()

        return self.alpha * ce + self.beta * rce


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
