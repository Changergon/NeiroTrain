import os
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path


def analyze_class_distribution(data_dir):
    """
    Анализирует распределение классов в датасете и выводит гистограмму + статистику.

    Args:
        data_dir (str/Path): Путь к директории с данными (в формате ImageFolder)
    """
    # Собираем статистику
    class_counts = defaultdict(int)
    class_names = sorted(os.listdir(data_dir))

    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):
            count = len(os.listdir(class_dir))
            class_counts[class_name] = count

    # Сортируем по количеству изображений (по убыванию)
    sorted_counts = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    classes, counts = zip(*sorted_counts)

    # Вывод в консоль
    print("\n📊 Распределение классов:")
    print(f"Всего классов: {len(class_counts)}")
    print(f"Всего изображений: {sum(counts)}")
    print("\nКласс (кол-во изображений):")
    for class_name, count in sorted_counts:
        print(f"  {class_name:30} - {count:4} ({count / sum(counts) * 100:.1f}%)")

    # Построение гистограммы
    plt.figure(figsize=(12, 6))
    bars = plt.bar(classes, counts, color='skyblue')

    # Добавляем значения на столбцы
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height}',
                 ha='center', va='bottom', fontsize=9)

    plt.title('Распределение изображений по классам')
    plt.xlabel('Классы')
    plt.ylabel('Количество изображений')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Сохраняем и показываем график
    plt.savefig('class_distribution.png', dpi=300)
    print("\nГистограмма сохранена в 'class_distribution.png'")
    plt.show()


if __name__ == "__main__":
    # Пример использования (замените путь на свой)
    train_dir = Path("C:/Users/Дмитрий/Desktop/ECOMMERCE_PRODUCT_IMAGES/train")
    val_dir = Path("C:/Users/Дмитрий/Desktop/ECOMMERCE_PRODUCT_IMAGES/val")
    check_dir = Path("C:/Users/Дмитрий/Desktop/ECOMMERCE_PRODUCT_IMAGES/check")

    print("=" * 50)
    print("АНАЛИЗ ТРЕНИРОВОЧНОГО НАБОРА:")
    analyze_class_distribution(train_dir)

    print("\n" + "=" * 50)
    print("АНАЛИЗ ВАЛИДАЦИОННОГО НАБОРА:")
    analyze_class_distribution(val_dir)

    print("\n" + "=" * 50)
    print("АНАЛИЗ ТЕСТОВОГО НАБОРА:")
    analyze_class_distribution(check_dir)