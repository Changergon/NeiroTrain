import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay, f1_score
from tqdm import tqdm


def evaluate_with_f1(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Оценка с F1"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    f1 = f1_score(all_labels, all_preds, average='weighted')
    return f1


# --- Сохранение неверно классифицированных изображений ---
def save_misclassified_examples(model, loader, device, output_dir, num_samples=5):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    misclassified = []
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Сохранение неверно классифицированных"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            misclassified_mask = preds != labels
            misclassified.extend([(inputs[i], labels[i], preds[i]) for i in range(len(inputs)) if misclassified_mask[i]])

    for i, (img, label, pred) in enumerate(misclassified[:num_samples]):
        img = img.cpu().numpy().transpose((1, 2, 0)) * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        img = (img * 255).astype(np.uint8)
        Image.fromarray(img).save(f"{output_dir}/misclassified_{i}_label_{label.item()}_pred_{pred.item()}.png")

# --- Построение confusion matrix ---
def plot_confusion_matrix(conf_matrix, class_names, output_file):
    fig, ax = plt.subplots(figsize=(12, 10))  # Увеличиваем размер под количество классов

    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=class_names)
    disp.plot(cmap='Blues', ax=ax, colorbar=False)

    # Поворачиваем подписи по оси X
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(fontsize=9)

    plt.tight_layout()  # Автоматическая подгонка
    plt.savefig(output_file, bbox_inches='tight')  # Сохраняем с учётом всех элементов
    plt.close(fig)