import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'fer2013'))
from model import (extract_net, AlexNetFer, ResNet18Fer, ResNet50Fer,
                   MobileNetV2Fer, ShuffleNet2Fer)

# Target RAF-DB Order: [0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral]

# Map: FER+ channels -> RAF-DB channels
# FER+ Src: 0:neu, 1:hap, 2:sur, 3:sad, 4:ang, 5:dig, 6:fea, 7:con
# Array index aligns with RAF-DB (0 to 6)
FERPLUS_TO_RAFDB = [2, 6, 5, 1, 3, 4, 0]  # [0:sur<-2, 1:fea<-6, 2:dig<-5, 3:hap<-1, 4:sad<-3, 5:ang<-4, 6:neu<-0]

# Map: FER2013 channels -> RAF-DB channels
# FER2013 Src (Alphabetical): 0:ang, 1:dig, 2:fea, 3:hap, 4:neu, 5:sad, 6:sur
# Array index aligns with RAF-DB (0 to 6)
FER2013_TO_RAFDB = [6, 2, 1, 3, 5, 0, 4]  # [0:sur<-6, 1:fea<-2, 2:dig<-1, 3:hap<-3, 4:sad<-5, 5:ang<-0, 6:neu<-4]

CLASS_REMAP = {
    'Ours-Micro-FER+':  FERPLUS_TO_RAFDB,
    'Ours-Nano-FER+':   FERPLUS_TO_RAFDB,
    'Ours-Tiny-FER+':   FERPLUS_TO_RAFDB,
    'Ours-Small-FER+':  FERPLUS_TO_RAFDB,

    'Ours-Micro':  FER2013_TO_RAFDB,
    'Ours-Nano':   FER2013_TO_RAFDB,
    'Ours-Tiny':   FER2013_TO_RAFDB,
    'Ours-Small':  FER2013_TO_RAFDB,
    'AlexNet':     FER2013_TO_RAFDB,
    'ResNet18':    FER2013_TO_RAFDB,
    'ResNet50':    FER2013_TO_RAFDB,
    'MobileNetV2': FER2013_TO_RAFDB,
    'ShuffleNetV2': FER2013_TO_RAFDB,
}


def _get_classifier_linear(model):
    for attr in ('classifier', 'fc'):
        mod = getattr(model, attr, None)
        if mod is None:
            continue
        if isinstance(mod, nn.Linear):
            return attr, mod
        if isinstance(mod, nn.Sequential):
            for i in range(len(mod) - 1, -1, -1):
                if isinstance(mod[i], nn.Linear):
                    return attr, mod[i], i
    raise RuntimeError("Cannot locate final nn.Linear in model")


def _set_classifier_linear(model, new_linear):
    result = _get_classifier_linear(model)
    if len(result) == 2:       
        attr, _ = result
        setattr(model, attr, new_linear)
    else:                     
        attr, _, idx = result
        getattr(model, attr)[idx] = new_linear


def remap_classifier(model, target_indices, target_num_classes):
    _, old_linear = _get_classifier_linear(model)[:2]
    old_weight = old_linear.weight.data
    old_bias = old_linear.bias.data
    in_features = old_weight.shape[1]

    new_linear = nn.Linear(in_features, target_num_classes)
    new_linear.weight.data = old_weight[target_indices].clone()
    new_linear.bias.data = old_bias[target_indices].clone()
    if old_weight.device.type != 'cpu':
        new_linear = new_linear.to(old_weight.device)
    _set_classifier_linear(model, new_linear)


def _num_classes_from_ckpt(state_dict):
    for k in ('classifier.weight', 'fc.weight'):
        if k in state_dict:
            return state_dict[k].shape[0]
    for k in reversed(list(state_dict.keys())):
        if k.endswith('.weight') and state_dict[k].ndim == 2:
            return state_dict[k].shape[0]
    return None


class RAFDBDataset(Dataset):
    def __init__(self, image_dir, label_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.samples = []
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 2:
                    continue
                fname, label = parts
                if 'test_' not in fname:
                    continue
                label = int(label) - 1
                img_name = fname.replace('.jpg', '_aligned.jpg')
                img_path = os.path.join(image_dir, img_name)
                if os.path.exists(img_path):
                    self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('L')
        if self.transform:
            img = self.transform(img)
        return img, label


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc='  Eval', leave=False):
            imgs = imgs.to(device)
            out = model(imgs)
            if isinstance(out, tuple):
                out = out[1]
            preds = out.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    acc = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds,
                                   target_names=['Surprise','Fear','Disgust','Happiness',
                                                 'Sadness','Anger','Neutral'],
                                   digits=4, output_dict=True, zero_division=0)
    return acc, report['macro avg']['f1-score']


def main():
    # device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    device = torch.device('cpu') 
    print(f"Device: {device}")

    transform = transforms.Compose([
        transforms.Resize(48),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,)),
    ])

    image_dir = os.path.join(os.path.dirname(__file__), 'Image', 'aligned')
    label_file = os.path.join(os.path.dirname(__file__), 'EmoLabel',
                              'list_patition_label.txt')
    dataset = RAFDBDataset(image_dir, label_file, transform)
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=4)
    print(f"RAF-DB test samples: {len(dataset)}")

    fer2013_results = os.path.join(os.path.dirname(__file__), '..', 'fer2013',
                                   'results')
    ferplus_results = os.path.join(os.path.dirname(__file__), '..', 'ferplus',
                                   'results')

    TARGET_CLASSES = 7

    checkpoints = {
        'Ours-Micro':  os.path.join(fer2013_results, 'SwiftResNet_micro_20260526_230046', 'best_model.pth'),
        'Ours-Nano':   os.path.join(fer2013_results, 'SwiftResNet_nano_20260526_191933', 'best_model.pth'),
        'Ours-Tiny':   os.path.join(fer2013_results, 'SwiftResNet_tiny_20260527_175630', 'best_model.pth'),
        'Ours-Small':  os.path.join(fer2013_results, 'SwiftResNet_small_20260528_084007', 'best_model.pth'),
        'AlexNet':     os.path.join(fer2013_results, 'AlexNet_None_20260529_210613', 'best_model.pth'),
        'ResNet18':    os.path.join(fer2013_results, 'ResNet18_None_20260529_182048', 'best_model.pth'),
        'ResNet50':    os.path.join(fer2013_results, 'ResNet50_None_20260530_163033', 'best_model.pth'),
        'MobileNetV2': os.path.join(fer2013_results, 'MobileNetV2_None_20260529_171412', 'best_model.pth'),
        'ShuffleNetV2': os.path.join(fer2013_results, 'ShuffleNet2_None_20260529_163419', 'best_model.pth'),
        'Ours-Micro-FER+':  os.path.join(ferplus_results, 'SwiftResNet_micro_20260530_201856', 'best_model.pth'),
        'Ours-Nano-FER+':   os.path.join(ferplus_results, 'SwiftResNet_nano_20260530_210953', 'best_model.pth'),
        'Ours-Tiny-FER+':   os.path.join(ferplus_results, 'SwiftResNet_tiny_20260530_225234', 'best_model.pth'),
        'Ours-Small-FER+':  os.path.join(ferplus_results, 'SwiftResNet_small_20260531_085828', 'best_model.pth'),
    }

    # Base kwargs — num_classes will be adjusted per checkpoint at load time
    model_builders = {
        'Ours-Micro':  (extract_net, {'scale': 'micro'}),
        'Ours-Nano':   (extract_net, {'scale': 'nano'}),
        'Ours-Tiny':   (extract_net, {'scale': 'tiny'}),
        'Ours-Small':  (extract_net, {'scale': 'small'}),
        'AlexNet':     (AlexNetFer, {}),
        'ResNet18':    (ResNet18Fer, {}),
        'ResNet50':    (ResNet50Fer, {}),
        'MobileNetV2': (MobileNetV2Fer, {'input_size': 48, 'width_mult': 1.1}),
        'ShuffleNetV2': (ShuffleNet2Fer, {'input_size': 48, 'net_type': 1}),
        'Ours-Micro-FER+':  (extract_net, {'scale': 'micro'}),
        'Ours-Nano-FER+':   (extract_net, {'scale': 'nano'}),
        'Ours-Tiny-FER+':   (extract_net, {'scale': 'tiny'}),
        'Ours-Small-FER+':  (extract_net, {'scale': 'small'}),
    }

    results = []
    for name, ckpt_path in checkpoints.items():
        if not os.path.exists(ckpt_path):
            print(f"SKIP {name}: checkpoint not found")
            continue

        # --- determine checkpoint num_classes & build model accordingly ---
        state = torch.load(ckpt_path, map_location='cpu')
        ckpt_classes = _num_classes_from_ckpt(state)
        if ckpt_classes is None:
            print(f"SKIP {name}: unable to determine num_classes from checkpoint")
            continue

        builder, kwargs = model_builders[name]
        kwargs_with_cls = dict(kwargs)
        kwargs_with_cls['num_classes'] = ckpt_classes
        # MobileNetV2 uses 'n_class' instead of 'num_classes'
        if builder is MobileNetV2Fer:
            kwargs_with_cls['n_class'] = ckpt_classes
            kwargs_with_cls.pop('num_classes', None)

        model = builder(**kwargs_with_cls).to(device)
        model.load_state_dict(state)

        # --- remap classifier when checkpoint class order ≠ target ---
        remap_indices = CLASS_REMAP.get(name)
        if remap_indices is not None:
            remap_classifier(model, remap_indices, TARGET_CLASSES)

        acc, f1 = evaluate(model, loader, device)
        results.append((name, acc, f1))
        print(f"{name:20s} | Acc={acc:.4f} ({acc*100:.2f}%) | F1={f1:.4f}")

    print("\n=== SUMMARY ===")
    results.sort(key=lambda x: x[1], reverse=True)
    for name, acc, f1 in results:
        print(f"{name:20s} | {acc*100:6.2f}% | F1={f1:.4f}")


if __name__ == '__main__':
    main()
