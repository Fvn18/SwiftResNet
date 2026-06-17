import json
import os
from typing import Dict, Any

DEFAULTS: Dict[str, Any] = {
    # ==================== Base Settings ====================
    'dataset': 'ferplus',
    'device': 'auto',
    'num_workers': 8,
    'seed': 67,
    'use_amp': True,

    # ==================== Model Settings ====================
    'model': 'SwiftResNet',
    'scale': 'small',                    # micro / nano / tiny / small
    'num_classes': 8,

    # ==================== Data Paths ====================
    'train_path': 'ferplus.csv',
    'val_path': 'ferplus.csv',
    'test_path': 'ferplus.csv',

    # ==================== Training Hyperparameters ====================
    'batch_size': 256,
    'epochs': 300,
    'lr': 0.14,
    'weight_decay': 7e-5,
    'patience': 300,
    'gradient_clip': 1.0,

    # ==================== Optimizer ====================
    'optimizer': 'sgd',
    'momentum': 0.9,
    'betas': (0.9, 0.999),          

    # ==================== Learning Rate Scheduler ====================
    'scheduler': 'cosine',                
    'use_warmup': True,
    'warmup_epochs': 10,
    'cosine_eta_min': 3e-6,
    'cosine_T_max': 300,                 

    'plateau_epochs': 10,

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
    print("FERPLUS Training Configuration (Production Ready)")
    print("=" * 70)
    for key, value in sorted(config.items()):
        print(f"{key:30s}: {value}")
    print("=" * 70 + "\n")


def store_config(config: Dict[str, Any], config_path: str) -> None:
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
