import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from utils.losses import mixup_criterion


# --- EarlyStopping ---
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta  # Минимальный значимый прирост
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:  # Учитываем min_delta
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


# --- MixUp ---
def mixup_data(x, y, alpha=1.0):
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def cutmix_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(x.size()[0]).to(x.device)
    y_a, y_b = y, y[rand_index]

    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size(-1) * x.size(-2)))
    return x, y_a, y_b, lam

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


# --- Обучение ---
def train_one_epoch(model, loader, criterion, optimizer, device,
                    use_mixup=False, mixup_prob=0.5, mixup_alpha=1.0,
                    use_cutmix=False, cutmix_prob=0.5, cutmix_alpha=1.0):
    model.train()
    total_loss = 0.0

    for inputs, labels in tqdm(loader, desc="Обучение"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        # Применяем аугментации в соответствии с параметрами
        if use_cutmix and np.random.rand() < cutmix_prob:
            inputs, targets_a, targets_b, lam = cutmix_data(inputs, labels, cutmix_alpha)
            outputs = model(inputs)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        elif use_mixup and np.random.rand() < mixup_prob:
            inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, mixup_alpha)
            outputs = model(inputs)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)




# --- Оценка ---
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Валидация"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return 100 * correct / total

# --- Оценка по классам ---
def evaluate_per_class(model, loader, device):
    model.eval()
    num_samples = len(loader.dataset)
    num_classes = len(loader.dataset.classes)

    all_preds = torch.zeros(num_samples, dtype=torch.long).to(device)
    all_labels = torch.zeros(num_samples, dtype=torch.long).to(device)

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(tqdm(loader, desc="Оценка по классам")):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            start = i * loader.batch_size
            end = start + labels.size(0)
            all_preds[start:end] = preds
            all_labels[start:end] = labels

    cm = confusion_matrix(all_labels.cpu(), all_preds.cpu(), labels=range(num_classes))
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)

    return cm, per_class_accuracy