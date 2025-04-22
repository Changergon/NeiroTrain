import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

from models.improved_cnn import ImprovedCNN
from utils.data_loading import remove_empty_folders, load_datasets
from utils.losses import FocalLoss
from utils.metrics import evaluate_with_f1, plot_confusion_matrix
from utils.training import EarlyStopping, train_one_epoch, evaluate, evaluate_per_class


# --- Основной цикл ---
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(args.log_dir)

    # --- Информация о системе и параметрах ---
    print(f"\n🖥️ Используемое устройство: {device}")
    if torch.cuda.is_available():
        print(f"📦 Видеокарта: {torch.cuda.get_device_name(0)}")

    print("\n🔧 Параметры запуска:")
    print(f"    📁 Train Dir:       {args.train_dir}")
    print(f"    📁 Val Dir:         {args.val_dir}")
    print(f"    📁 Check Dir:       {args.check_dir}")
    print(f"    💾 Model Path:      {args.model_path}")
    print(f"    📦 Batch Size:      {args.batch_size}")
    print(f"    🔁 Epochs:          {args.epochs}")
    print(f"    🚀 Learning Rate:   {args.lr}")
    print(f"    ⚖️ Weight Decay:    {args.weight_decay}")

    # Параметры модели
    print("\n🧠 Параметры модели:")
    print(f"    🎯 Loss Function:   {'Focal Loss' if args.loss == 'focal' else 'CrossEntropy'}")
    if args.loss == 'focal':
        print(f"    🔍 Focal Gamma:     {args.focal_gamma}")
    print(f"    🧩 Использовать CBAM: {'Да' if args.use_cbam else 'Нет'}")
    if args.use_cbam:
        print(f"    🔢 CBAM Ratio:      {args.cbam_ratio}")
        print(f"    📏 CBAM Kernel:     {args.cbam_kernel_size}")
    print(f"    🧊 Использовать BatchNorm: {'Да' if args.use_batchnorm else 'Нет'}")

    # Параметры регуляризации
    print("\n🛡️ Параметры регуляризации:")
    print(f"    🧱 Использовать DropBlock: {'Да' if args.use_dropblock else 'Нет'}")
    if args.use_dropblock:
        print(f"    🎲 Drop Probability:  {args.drop_prob}")
        print(f"    📦 Block Size:        {args.block_size}")

    # Параметры аугментаций
    print("\n🎨 Параметры аугментаций:")
    print(f"    🧪 MixUp:           {'Включен' if args.use_mixup else 'Отключен'}")
    if args.use_mixup:
        print(f"    🎲 MixUp Prob:      {args.mixup_prob}")
        print(f"    α (alpha) MixUp:    {args.mixup_alpha}")
    print(f"    🩹 CutMix:          {'Включен' if args.use_cutmix else 'Отключен'}")
    if args.use_cutmix:
        print(f"    🎲 CutMix Prob:      {args.cutmix_prob}")
        print(f"    α (alpha) CutMix:   {args.cutmix_alpha}")

    # Параметры обучения
    print("\n⚙️ Параметры обучения:")
    print(f"    ⚙️ Оптимизатор:     {args.optimizer.upper()}")
    if args.optimizer == 'sgd':
        print(f"    🏃 Momentum:        {args.momentum}")
    print(f"    📉 Scheduler:       {args.scheduler.capitalize()}")
    print(f"    ⏳ Patience:         {args.patience}")
    print(f"    🏁 Min LR:          {args.min_lr}")

    # Параметры логирования
    print("\n📊 Параметры логирования:")
    print(f"    📂 Log Dir:         {args.log_dir}")
    print(f"    ❌ Сохранять ошибки: {'Да' if args.save_misclassified else 'Нет'}")
    if args.save_misclassified:
        print(f"    📁 Директория ошибок: {args.misclassified_dir}")
        print(f"    #️⃣ Кол-во примеров:  {args.num_misclassified}")




    # Проверка и загрузка данных
    remove_empty_folders(args.train_dir)
    remove_empty_folders(args.val_dir)

    train_dataset, val_dataset, check_dataset = load_datasets(args.train_dir, args.val_dir, args.check_dir)
    num_classes = len(train_dataset.classes)

    class_counts = np.array(
        [len(os.listdir(os.path.join(args.train_dir, class_name))) for class_name in train_dataset.classes])
    class_weights = 1.0 / class_counts
    sample_weights = np.array([class_weights[train_dataset.targets[i]] for i in range(len(train_dataset))])
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    check_loader = DataLoader(check_dataset, batch_size=args.batch_size)

    if args.balance_classes:
        # Увеличиваем вес редких классов
        class_weights = 1.0 / (class_counts + 1e-6)  # +1e6 чтобы избежать деления на 0
        sample_weights = np.array([class_weights[y] for y in train_dataset.targets])
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)


    model = ImprovedCNN(
        num_classes=num_classes,
        use_cbam=args.use_cbam,
        cbam_ratio=args.cbam_ratio,
        cbam_kernel_size=args.cbam_kernel_size,
        use_dropblock=args.use_dropblock,
        drop_prob=args.drop_prob,
        block_size=args.block_size,
        use_batchnorm=args.use_batchnorm
    ).to(device)



    # --- Выбор функции потерь ---
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    if args.loss == 'focal':
        # Усиливаем фокус на малых классах
        gamma = 2.0  # Увеличиваем с 1.5 до 2.0
        criterion = FocalLoss(alpha=class_weights_tensor, gamma=gamma)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    early_stopping = EarlyStopping(patience=5)

    best_accuracy = 0.0
    losses, accuracies = [], []

    for epoch in range(args.epochs):
        print(f"\n{'=' * 25} Эпоха {epoch + 1} / {args.epochs} {'=' * 25}")
        avg_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            use_mixup=args.use_mixup,
            mixup_alpha=args.mixup_alpha,
            use_cutmix=args.use_cutmix,
            cutmix_prob=args.cutmix_prob,
            cutmix_alpha=args.cutmix_alpha
        )

        accuracy = evaluate(model, val_loader, device)
        losses.append(avg_loss)
        accuracies.append(accuracy)
        train_f1 = evaluate_with_f1(model, train_loader, device)
        val_f1 = evaluate_with_f1(model, val_loader, device)
        writer.add_scalar("Loss/train", avg_loss, epoch)
        writer.add_scalar("Accuracy/val", accuracy, epoch)
        writer.add_scalar("F1/train", train_f1, epoch)
        writer.add_scalar("F1/val", val_f1, epoch)



        # Сохранение матрицы ошибок в runs/matrix
        output_dir = Path("runs/matrix")
        output_dir.mkdir(parents=True, exist_ok=True)  # Создание директории, если она не существует


        scheduler.step(accuracy)
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar("LR", current_lr, epoch)

        print(f"📉 Средняя потеря (loss):        {avg_loss:.4f}")
        print(f"✅ Точность на валидации:        {accuracy:.2f}%")
        print(f"🎯 F1-score на валидации:        {val_f1:.4f}")
        print(f"🔁 Текущий learning rate:        {current_lr:.6f}")

        # Обновление EarlyStopping
        early_stopping(accuracy)  # Передаем текущую точность в EarlyStopping

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model, args.model_path)
            print("💾 Лучшая модель сохранена!")

        if early_stopping.early_stop:
            print("⏹️ Ранняя остановка.")
            break
        print("=" * 60)

    # Оценка на check-сете по завершению всех эпох
    check_accuracy = evaluate(model, check_loader, device)
    print(f"✅ Точность на проверочном наборе (check_loader):        {check_accuracy:.2f}%")
    writer.add_scalar("Accuracy/check", check_accuracy, args.epochs)
    # --- Финальная оценка по классам ---
    conf_matrix, class_accuracies = evaluate_per_class(model, check_loader, device)
    plot_confusion_matrix(conf_matrix, check_dataset.classes, Path("runs/matrix") / "confusion_matrix_final.png")

    print(f"\n📊 Точность по классам (на check-сете):")
    for cls_name, acc in zip(check_dataset.classes, class_accuracies):
        print(f"    📦 {cls_name:25} — {acc * 100:.2f}%")

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ImprovedCNN with CBAM on E-commerce Product Images")

    # 1. Основные параметры данных и обучения
    parser.add_argument('--train_dir', type=Path, default=Path("C:\\Users\\Дмитрий\\Desktop\\ECOMMERCE_PRODUCT_IMAGES\\train"),
                        help="Путь к тренировочному датасету")
    parser.add_argument('--val_dir', type=Path, default=Path("C:\\Users\\Дмитрий\\Desktop\\ECOMMERCE_PRODUCT_IMAGES\\val"),
                        help="Путь к валидационному датасету")
    parser.add_argument('--check_dir', type=Path, default=Path("C:\\Users\\Дмитрий\\Desktop\\ECOMMERCE_PRODUCT_IMAGES\\check"),
                        help="Путь к тестовому датасету")
    parser.add_argument('--model_path', type=str, default="best_model.pth",
                        help="Путь для сохранения лучшей модели")
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Размер батча")
    parser.add_argument('--epochs', type=int, default=100,
                        help="Количество эпох обучения")
    parser.add_argument('--lr', type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument('--weight_decay', type=float, default=0.001,
                        help="Weight decay для оптимизатора")

    # 2. Параметры модели
    parser.add_argument('--use_cbam', action='store_true', default=True,
                        help="Использовать CBAM модули")
    parser.add_argument('--cbam_ratio', type=int, default=16,
                        help="Ratio для ChannelAttention в CBAM")
    parser.add_argument('--cbam_kernel_size', type=int, default=7,
                        help="Kernel size для SpatialAttention в CBAM")

    # 3. Параметры регуляризации
    parser.add_argument('--use_dropblock', action='store_true', default=True,
                        help="Использовать DropBlock")
    parser.add_argument('--drop_prob', type=float, default=0.1,
                        help="Вероятность дропа для DropBlock")
    parser.add_argument('--block_size', type=int, default=5,
                        help="Размер блока для DropBlock")
    parser.add_argument('--use_batchnorm', action='store_true', default=True,
                        help="Использовать BatchNorm")

    # 4. Параметры аугментаций
    parser.add_argument('--use_mixup', action='store_true', default=True,
                        help="Использовать MixUp аугментацию")
    parser.add_argument('--mixup_prob', type=float, default=0.3,
                        help="Вероятность применения MixUp")
    parser.add_argument('--mixup_alpha', type=float, default=1.0,
                        help="Alpha параметр для MixUp")
    parser.add_argument('--use_cutmix', action='store_true', default=True,
                        help="Использовать CutMix аугментацию")
    parser.add_argument('--cutmix_prob', type=float, default=0.3,
                        help="Вероятность применения CutMix")
    parser.add_argument('--cutmix_alpha', type=float, default=1.0,
                        help="Alpha параметр для CutMix")

    # 5. Параметры функции потерь
    parser.add_argument('--loss', type=str, choices=['ce', 'focal'], default='focal',
                        help="Тип функции потерь")
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help="Gamma параметр для Focal Loss")

    # 6. Параметры обучения
    parser.add_argument('--optimizer', type=str, choices=['adam', 'sgd'], default='adam',
                        help="Тип оптимизатора")
    parser.add_argument('--momentum', type=float, default=0.9,
                        help="Momentum для SGD")
    parser.add_argument('--scheduler', type=str, choices=['plateau', 'step', 'cosine'], default='plateau',
                        help="Тип шедулера для LR")
    parser.add_argument('--patience', type=int, default=10,
                        help="Patience для EarlyStopping и ReduceLROnPlateau")
    parser.add_argument('--min_lr', type=float, default=0.0000000001,
                        help="Минимальный learning rate")
    # В разделе "Параметры обучения" добавьте:
    parser.add_argument('--balance_classes', action='store_true', default=True,
                        help="Балансировка классов через WeightedRandomSampler")


    # 7. Параметры логирования
    parser.add_argument('--log_dir', type=str, default="runs",
                        help="Директория для логов TensorBoard")
    parser.add_argument('--save_misclassified', action='store_true', default=True,
                        help="Сохранять примеры с ошибками классификации")
    parser.add_argument('--misclassified_dir', type=str, default="misclassified",
                        help="Директория для сохранения ошибочных примеров")
    parser.add_argument('--num_misclassified', type=int, default=5,
                        help="Количество сохраняемых ошибочных примеров")

    args = parser.parse_args()

    # Проверка аргументов
    if args.use_cutmix and args.use_mixup:
        print("Предупреждение: И CutMix, и MixUp включены одновременно. Это может быть избыточно.")

    main(args)