import json
import os
from typing import Dict, Any

DEFAULTS: Dict[str, Any] = {
    # ==================== Base Settings ====================
    'dataset': 'fer2013',
    'device': 'auto',
    'num_workers': 8,
    'seed': 67,
    'use_amp': True,

    # ==================== Model Settings ====================
    'model': 'ResNet50',                    # SwiftResNet / AlexNet / ResNet50 / ResNet18 / ShuffleNet2 / MobileNetV2
    'scale': 'small',                    # micro / nano / tiny / small
    'num_classes': 7,

    # ==================== Data Paths ====================
    'train_path': 'fer2013/train',
    'val_path': 'fer2013/val',
    'test_path': 'fer2013/test',

    # ==================== Training Hyperparameters ====================
    'batch_size': 256,
    'epochs': 300,
    'lr': 0.14,
    'weight_decay': 9e-5,
    'patience': 300,
    'gradient_clip': 1.0,

    # ==================== Optimizer ====================
    'optimizer': 'sgd',
    'momentum': 0.9,
    'betas': (0.9, 0.999),           # Only used for Adam. Ignored if optimizer is not Adam.

    # ==================== Learning Rate Scheduler ====================
    'scheduler': 'cosine',                 # cosine / step 
    'use_warmup': True,
    'warmup_epochs': 10,
    'cosine_eta_min': 3e-6,
    'cosine_T_max': 300,                 

    'plateau_epochs': 10,

    'step_size': 12,
    'step_gamma': 0.82,


    # ==================== Loss Function ====================
    'loss_type': 'sce',                 # sce / ce 
    'sce_alpha': 1.0,
    'sce_beta': 0.3,
    'label_smoothing': 0.1,

    # ==================== Data Augmentation ====================
    'mixup_alpha': 0.1,
    'cutmix_alpha': 0.45,

    # ==================== Reproducibility ====================
    'cudnn_deterministic': True,
    'cudnn_benchmark': False,

}


def get_defaults() -> Dict[str, Any]:
    return DEFAULTS.copy()


def print_config(config: Dict[str, Any]) -> None:
    print("\n" + "=" * 70)
    print("FER2013 Training Configuration (Production Ready)")
    print("=" * 70)
    for key, value in sorted(config.items()):
        print(f"{key:30s}: {value}")
    print("=" * 70 + "\n")


def store_config(config: Dict[str, Any], config_path: str) -> None:
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
