import os
import logging
import argparse
import random
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
from tqdm.auto import tqdm

from config import get_defaults, print_config, store_config
from utils import set_seed, mixup_data, cutmix_data, mixup_criterion
from dataset import load_data
from model import extract_net

class FERPlusTrainer:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.device = self.__get_device()
        self.result_dir = self.__create_result_dir()
        self.__setup_logging()

        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.train_losses, self.train_accs = [], []
        self.val_losses, self.val_accs = [], []
        self.learning_rates = []

        self.use_amp = config.get('use_amp', False) and self.device.type == 'cuda'
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)

        torch.backends.cudnn.deterministic = config.get('cudnn_deterministic', True)
        torch.backends.cudnn.benchmark = config.get('cudnn_benchmark', False)

        print_config(config)
        store_config(config, os.path.join(self.result_dir, 'config.json'))

    def __get_device(self) -> torch.device:
        device_type = 'cuda' if (self.config['device'] == 'auto' and torch.cuda.is_available()) else 'cpu'
        return torch.device(device_type)

    def __create_result_dir(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        scale = self.config.get('scale', 'tiny')
        path = os.path.join('results', f"SwiftResNet_{scale}_{timestamp}")
        os.makedirs(path, exist_ok=True)
        return path

    def __setup_logging(self) -> None:
        log_file = os.path.join(self.result_dir, 'training.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(levelname)s | %(message)s',
            handlers=[logging.FileHandler(log_file, encoding='utf-8'), logging.StreamHandler()]
        )
        self.logger = logging.getLogger(__name__)

    def __prepare_data(self) -> None:
        seed = self.config.get('seed', 42)
        common_kwargs = {
            "batch_size": self.config['batch_size'],
            "num_workers": self.config['num_workers'],
            "seed": seed
        }
        self.train_loader = load_data(path=self.config['train_path'], mode='train', **common_kwargs)
        self.val_loader = load_data(path=self.config['val_path'], mode='val', **common_kwargs)
        self.test_loader = load_data(path=self.config['test_path'], mode='test', **common_kwargs)

        self.class_names = self.train_loader.dataset.classes_8
        self.logger.info(f"FERPlus Data Ready | Train: {len(self.train_loader.dataset)} | Val: {len(self.val_loader.dataset)} | Test: {len(self.test_loader.dataset)}")

    def __build_engine(self) -> None:
        self.model = extract_net(
            scale=self.config.get('scale', 'tiny'),
            num_classes=self.config.get('num_classes', 10)
        ).to(self.device)

        self.criterion = nn.KLDivLoss(reduction='batchmean')

        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.config['lr'],
            momentum=self.config.get('momentum', 0.9),
            weight_decay=self.config['weight_decay'],
            nesterov=True
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.get('cosine_t_max', self.config['epochs']),
            eta_min=self.config.get('cosine_eta_min', 1e-6)
        )

    def __apply_augmentation(self, images: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        mixup_alpha = self.config.get('mixup_alpha', 0.0)
        cutmix_alpha = self.config.get('cutmix_alpha', 0.0)

        if (mixup_alpha > 0 or cutmix_alpha > 0) and random.random() < 0.5:
            use_mixup = random.random() < 0.5 if (mixup_alpha > 0 and cutmix_alpha > 0) else mixup_alpha > 0
            aug_fn = mixup_data if use_mixup else cutmix_data
            alpha = mixup_alpha if use_mixup else cutmix_alpha
            return aug_fn(images, labels, alpha=alpha, device=self.device)

        return images, labels, labels, 1.0

    def __compute_kl_loss(self, logits: torch.Tensor, target_a: torch.Tensor, target_b: torch.Tensor, lam: float) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        if lam != 1.0:
            loss = lam * self.criterion(log_probs, target_a) + (1.0 - lam) * self.criterion(log_probs, target_b)
        else:
            loss = self.criterion(log_probs, target_a)
        return loss

    def __optimization_step(self, loss: torch.Tensor) -> None:
        self.optimizer.zero_grad(set_to_none=True)
        if self.use_amp:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

    def __train_one_epoch(self, epoch: int) -> Tuple[float, float]:
        self.model.train()
        total_loss, total_correct, total_count = 0.0, 0, 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} Training", leave=False)
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)

            images, t_a, t_b, lam = self.__apply_augmentation(images, labels)

            with torch.amp.autocast('cuda', enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.__compute_kl_loss(outputs, t_a, t_b, lam)

            self.__optimization_step(loss)

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_count += batch_size

            _, predicted = torch.max(outputs, 1)
            idx_a, idx_b = t_a.argmax(dim=1), t_b.argmax(dim=1)
            total_correct += (lam * predicted.eq(idx_a).sum() + (1.0 - lam) * predicted.eq(idx_b).sum()).item()

            pbar.set_postfix({'loss': f"{total_loss/total_count:.4f}"})

        return total_loss / total_count, total_correct / total_count

    @torch.no_grad()
    def __evaluate(self, loader: DataLoader) -> Tuple[float, float, List[int], List[int]]:
        self.model.eval()
        total_loss, total_correct, total_count = 0.0, 0, 0
        all_labels, all_preds = [], []

        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)

            with torch.amp.autocast('cuda', enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.criterion(F.log_softmax(outputs, dim=1), labels)

            total_loss += loss.item() * images.size(0)
            total_count += images.size(0)

            _, predicted = torch.max(outputs, 1)
            target_idx = labels.argmax(dim=1)

            total_correct += predicted.eq(target_idx).sum().item()
            all_labels.extend(target_idx.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

        return total_loss / total_count, total_correct / total_count, all_labels, all_preds

    def __validate_epoch(self) -> Tuple[float, float, List[int], List[int]]:
        return self.__evaluate(self.val_loader)

    def __update_lr(self, epoch: int) -> None:
        warmup_epochs = self.config.get('warmup_epochs', 10)
        plateau_epochs = self.config.get('plateau_epochs', 1)

        if epoch < warmup_epochs:
            new_lr = self.config['lr'] * (epoch + 1) / warmup_epochs
            for g in self.optimizer.param_groups: g['lr'] = new_lr
        elif epoch >= warmup_epochs + plateau_epochs:
            self.scheduler.step()

        self.learning_rates.append(self.optimizer.param_groups[0]['lr'])

    def train(self) -> None:
        self.__prepare_data()
        self.__build_engine()

        patience = 0
        for epoch in range(self.config['epochs']):
            train_loss, train_acc = self.__train_one_epoch(epoch)
            val_loss, val_acc, _, _ = self.__validate_epoch()
            self.__update_lr(epoch)

            self.train_losses.append(train_loss); self.train_accs.append(train_acc)
            self.val_losses.append(val_loss); self.val_accs.append(val_acc)

            self.logger.info(f"Epoch {epoch+1:03d} | LR: {self.learning_rates[-1]:.6f} | "
                             f"Train Loss/Acc: {train_loss:.4f}/{train_acc:.4f} | "
                             f"Val Loss/Acc: {val_loss:.4f}/{val_acc:.4f}")

            if val_acc > self.best_val_acc:
                self.best_val_acc, self.best_epoch = val_acc, epoch + 1
                patience = 0
                torch.save(self.model.state_dict(), os.path.join(self.result_dir, 'best_model.pth'))
                self.logger.info(f" Saved Best Model: {val_acc:.4f}")
            else:
                patience += 1
                if patience >= self.config.get('patience', 30): break

        self.__final_evaluation()

    def __final_evaluation(self) -> None:
        self.model.load_state_dict(torch.load(os.path.join(self.result_dir, 'best_model.pth'), map_location=self.device))

        test_loss, test_acc, labels, preds = self.__evaluate(self.test_loader)

        self.logger.info(f"\nTest Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

        report = classification_report(labels, preds, target_names=self.class_names, digits=4)
        with open(os.path.join(self.result_dir, 'classification_report.txt'), 'w') as f:
            f.write(report)

        self.logger.info(f"\nFinal Classification Report:\n{report}")
        self.__plot_results()
        self.__plot_confusion_matrix(labels, preds)

    def __plot_results(self) -> None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        ax1.plot(self.train_losses, label='Train'); ax1.plot(self.val_losses, label='Val')
        ax1.set_title('KL Divergence Loss'); ax1.legend()

        ax2.plot(self.train_accs, label='Train'); ax2.plot(self.val_accs, label='Val')
        ax2.set_title('Accuracy (Hard Label Max)'); ax2.legend()

        plt.savefig(os.path.join(self.result_dir, 'training_curves.png'), dpi=150)
        plt.close()

    def __plot_confusion_matrix(self, labels: List[int], preds: List[int]) -> None:
        cm_result = confusion_matrix(labels, preds)
        norm_cm = cm_result.astype('float') / cm_result.sum(axis=1)[:, np.newaxis]

        plt.figure(figsize=(10, 8))
        sns.heatmap(norm_cm, annot=True, fmt='.2f', xticklabels=self.class_names, yticklabels=self.class_names, cmap='Blues')
        plt.title('Normalized Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(os.path.join(self.result_dir, 'confusion_matrix.png'), dpi=150)
        plt.close()

if __name__ == '__main__':
    config = get_defaults()
    if os.name == 'nt': mp.set_start_method('spawn', force=True)
    set_seed(config.get('seed', 42))
    FERPlusTrainer(config).train()
