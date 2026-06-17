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
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
from tqdm.auto import tqdm

from config import get_defaults, print_config, store_config
from utils import set_seed, SCELoss, mixup_data, cutmix_data, mixup_criterion
from dataset import load_data
from model import extract_net, AlexNetFer, ResNet50Fer, ShuffleNet2Fer, MobileNetV2Fer, ResNet18Fer


class Trainer:

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.device = self.__get_device()
        self.result_dir = self.__create_result_dir()
        self.__setup_logging()

        self.best_val_acc: float = 0.0
        self.best_epoch: int = 0
        self.train_losses, self.train_accs = [], []
        self.val_losses, self.val_accs = [], []
        self.learning_rates = []

        self.use_amp: bool = config.get('use_amp', False) and self.device.type == 'cuda'
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)

        torch.backends.cudnn.deterministic = config.get('cudnn_deterministic', True)
        torch.backends.cudnn.benchmark = config.get('cudnn_benchmark', False)

        print_config(config)
        store_config(config, os.path.join(self.result_dir, 'config.json'))

    def __get_device(self) -> torch.device:
        use_cuda = self.config['device'] == 'auto' and torch.cuda.is_available()
        return torch.device('cuda' if use_cuda else 'cpu')

    def __create_result_dir(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        scale = self.config.get('scale', 'tiny') if self.config.get('model', 'SwiftResNet') == 'SwiftResNet' else None
        model_name = self.config.get('model', 'SwiftResNet')
        result_dir = os.path.join('results', f"{model_name}_{scale}_{timestamp}")
        os.makedirs(result_dir, exist_ok=True)
        return result_dir

    def __setup_logging(self) -> None:
        log_file = os.path.join(self.result_dir, 'training.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(levelname)s | %(message)s',
            handlers=[logging.FileHandler(log_file, encoding='utf-8'), logging.StreamHandler()]
        )
        self.logger = logging.getLogger(__name__)

    def __prepare_datasets(self) -> None:
        seed = self.config.get('seed', 42)
        batch_size = self.config['batch_size']
        workers = self.config['num_workers']

        self.train_loader = load_data(self.config['train_path'], batch_size, workers, 'train', seed)
        self.val_loader = load_data(self.config['val_path'], batch_size, workers, 'val', seed)
        self.test_loader = load_data(self.config['test_path'], batch_size, workers, 'test', seed)

        self.class_names = self.train_loader.dataset.classes
        self.class_num_list = torch.unique(torch.tensor(self.train_loader.dataset.targets), return_counts=True)[1].tolist()

        self.logger.info(f"Dataset loaded → Train: {len(self.train_loader.dataset)} | Val: {len(self.val_loader.dataset)}")
        self.logger.info(f"Class counts: {self.class_num_list}")

    def __build_model(self) -> None:
        model_name = self.config.get('model', 'SwiftResNet')
        scale = self.config.get('scale', 'tiny') if model_name == 'SwiftResNet' else None
        num_classes = self.config.get('num_classes', 7)

        if model_name == 'SwiftResNet':
            self.model = extract_net(scale=scale, num_classes=num_classes).to(self.device)
        elif model_name == 'AlexNet':
            self.model = AlexNetFer(num_classes=num_classes).to(self.device)
        elif model_name == 'ResNet50':
            self.model = ResNet50Fer(num_classes=num_classes).to(self.device)
        elif model_name == 'ResNet18':
            self.model = ResNet18Fer(num_classes=num_classes).to(self.device)
        elif model_name == 'ShuffleNet2':
            self.model = ShuffleNet2Fer(num_classes=num_classes).to(self.device)
        elif model_name == 'MobileNetV2':
            self.model = MobileNetV2Fer(n_class=num_classes).to(self.device)

        total_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"Model: {model_name}-{scale} | Parameters: {total_params:,}")

    def __setup_training(self) -> None:
        num_classes = self.config.get('num_classes', 7)

        self.loss_type = self.config.get('loss_type', 'sce')
        loss_dict = {
            'sce': SCELoss(alpha=self.config['sce_alpha'], beta=self.config['sce_beta'], num_classes=num_classes, label_smoothing=self.config.get('label_smoothing', 0.1)),
            'ce': torch.nn.CrossEntropyLoss(label_smoothing=self.config.get('label_smoothing', 0.0)),
        }
        self.criterion = loss_dict[self.loss_type]
        self.logger.info(f"Loss function: {self.loss_type}")

        lr, wd = self.config['lr'], self.config['weight_decay']
        self.optimizer_type = self.config.get('optimizer', 'sgd')
        opt_dict = {
            'sgd': optim.SGD(self.model.parameters(), lr=lr, momentum=self.config.get('momentum', 0.9), weight_decay=wd, nesterov=True),
            'adam': optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd, betas=self.config.get('betas', (0.9, 0.999)))
        }
        self.optimizer = opt_dict[self.optimizer_type]
        self.logger.info(f"Optimizer: {self.optimizer_type.upper()}")

        scheduler_type = self.config.get('scheduler', 'cosine')
        warmup_epochs = self.config.get('warmup_epochs', 10)
        sched_dict = {
            'cosine': optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.config['cosine_T_max'], eta_min=self.config['cosine_eta_min']),
            'step': optim.lr_scheduler.StepLR(self.optimizer, step_size=self.config.get('step_size', 30), gamma=self.config.get('step_gamma', 0.1)),
        }
        self.scheduler = sched_dict[scheduler_type]

    def __warmup_lr(self, epoch: int) -> None:
        warmup_epochs = self.config.get('warmup_epochs', 10)
        if self.config.get('use_warmup', True) and epoch < warmup_epochs:
            warmup_lr = self.config['lr'] * float(epoch + 1) / float(warmup_epochs)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = warmup_lr

    def __apply_mixup_if_needed(self, images: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        mixup_alpha, cutmix_alpha = self.config.get('mixup_alpha', 0), self.config.get('cutmix_alpha', 0)

        if (mixup_alpha == 0 and cutmix_alpha == 0) or random.random() >= 0.5:
            return images, labels, labels, 1.0

        use_mixup = random.random() < 0.5 if (mixup_alpha > 0 and cutmix_alpha > 0) else (mixup_alpha > 0)
        augment_fn = mixup_data if use_mixup else cutmix_data
        alpha_val = mixup_alpha if use_mixup else cutmix_alpha

        return augment_fn(images, labels, alpha=alpha_val, device=self.device)

    def __compute_loss(self, outputs: torch.Tensor, labels: torch.Tensor,
                       targets_a: torch.Tensor, targets_b: torch.Tensor, mix_lambda: float, epoch: int) -> torch.Tensor:
        if mix_lambda != 1.0:
            return mixup_criterion(self.criterion, outputs, targets_a, targets_b, mix_lambda, epoch)
        return self.criterion(outputs, labels)

    def __optimization_step(self, loss: torch.Tensor) -> None:
        if self.use_amp:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
        else:
            loss.backward()

        if self.config.get('gradient_clip', 0) > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['gradient_clip'])

        if self.use_amp:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

    def __train_epoch(self, epoch: int) -> Tuple[float, float]:
        self.model.train()
        accumulated_loss, correct_predictions, total_samples = 0.0, 0, 0

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config['epochs']} [Train]", leave=False)

        for images, labels in progress_bar:
            images, labels = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
            batch_size = images.size(0)

            self.optimizer.zero_grad(set_to_none=True)

            images, targets_a, targets_b, mix_lambda = self.__apply_mixup_if_needed(images, labels)

            with torch.amp.autocast('cuda', enabled=self.use_amp):
                model_out = self.model(images)
                outputs = model_out[1] if isinstance(model_out, tuple) else model_out
                loss = self.__compute_loss(outputs, labels, targets_a, targets_b, mix_lambda, epoch)

            self.__optimization_step(loss)

            accumulated_loss += loss.item() * batch_size
            total_samples += batch_size
            _, predicted = torch.max(outputs, 1)

            if mix_lambda != 1.0:
                correct_predictions += (mix_lambda * predicted.eq(targets_a).sum() + (1.0 - mix_lambda) * predicted.eq(targets_b).sum()).item()
            else:
                correct_predictions += predicted.eq(labels).sum().item()

            progress_bar.set_postfix({'loss': f"{accumulated_loss/total_samples:.4f}", 'acc': f"{100*correct_predictions/total_samples:.2f}%"})

        return accumulated_loss / total_samples, correct_predictions / total_samples

    @torch.no_grad()
    def __evaluate(self, loader: DataLoader) -> Tuple[float, float, List[int], List[int]]:
        self.model.eval()
        accumulated_loss, correct_predictions, total_samples = 0.0, 0, 0
        all_labels, all_predictions = [], []

        for images, labels in loader:
            images, labels = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
            batch_size = labels.size(0)

            with torch.amp.autocast('cuda', enabled=self.use_amp):
                model_out = self.model(images)
                outputs = model_out[1] if isinstance(model_out, tuple) else model_out
                loss = self.criterion(outputs, labels)

            accumulated_loss += loss.item() * batch_size
            total_samples += batch_size
            _, predicted = torch.max(outputs, 1)
            correct_predictions += predicted.eq(labels).sum().item()

            all_labels.extend(labels.cpu().numpy().tolist())
            all_predictions.extend(predicted.cpu().numpy().tolist())

        return accumulated_loss / total_samples, correct_predictions / total_samples, all_labels, all_predictions

    def __validate_epoch(self) -> Tuple[float, float, List[int], List[int]]:
        return self.__evaluate(self.val_loader)

    def train(self) -> None:
        self.__prepare_datasets()
        self.__build_model()
        self.__setup_training()

        patience_counter = 0

        for epoch in range(self.config['epochs']):
            self.__warmup_lr(epoch)

            train_loss, train_acc = self.__train_epoch(epoch)
            val_loss, val_acc, _, _ = self.__validate_epoch()

            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)

            if epoch >= self.config.get('warmup_epochs', 10) + self.config.get('plateau_epochs', 0):
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)

            self.logger.info(f"Epoch {epoch+1:03d}/{self.config['epochs']} | LR: {current_lr:.6f} | "
                             f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                             f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

            if val_acc > self.best_val_acc:
                self.best_val_acc, self.best_epoch = val_acc, epoch + 1
                patience_counter = 0
                torch.save(self.model.state_dict(), os.path.join(self.result_dir, 'best_model.pth'))
                self.logger.info(f"[>>>]New Best Model Saved! Acc = {val_acc:.4f}")
            else:
                patience_counter += 1

            if patience_counter >= self.config.get('patience', 30):
                self.logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break

        self.__final_evaluation()

    def __final_evaluation(self) -> None:
        self.model.load_state_dict(torch.load(os.path.join(self.result_dir, 'best_model.pth'), map_location=self.device))

        test_loss, test_acc, all_labels, all_predictions = self.__evaluate(self.test_loader)

        self.logger.info(f"\nTest Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

        report = classification_report(all_labels, all_predictions, target_names=self.class_names, digits=4)
        with open(os.path.join(self.result_dir, 'classification_report.txt'), 'w') as f:
            f.write(report)

        self.logger.info(f"\nFinal Report:\n{report}")
        self.__plot_training_curves()
        self.__plot_confusion_matrix(all_labels, all_predictions)

    def __plot_training_curves(self) -> None:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        axes[0].plot(self.train_losses, label='Train')
        axes[0].plot(self.val_losses, label='Val')
        axes[0].set_title('Loss')
        axes[0].legend()
        axes[0].grid()

        axes[1].plot(self.train_accs, label='Train')
        axes[1].plot(self.val_accs, label='Val')
        axes[1].set_title('Accuracy')
        axes[1].legend()
        axes[1].grid()

        axes[2].plot(self.learning_rates)
        axes[2].set_title('Learning Rate')
        axes[2].set_yscale('log')
        axes[2].grid()

        plt.tight_layout()
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    config = get_defaults()
    if args.seed is not None:
        config['seed'] = args.seed

    if os.name == 'nt':
        mp.set_start_method('spawn', force=True)

    set_seed(config['seed'])
    Trainer(config).train()
