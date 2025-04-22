import os

from PIL import Image
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms


# --- Удаление пустых папок ---
def remove_empty_folders(root_dir: str):
    for dirpath, dirnames, _ in os.walk(root_dir):
        for dirname in dirnames:
            full_path = os.path.join(dirpath, dirname)
            if not os.listdir(full_path):
                os.rmdir(full_path)

# --- Безопасный загрузчик ---
def safe_pil_loader(path):
    image = Image.open(path)
    if image.mode in ('P', 'RGBA'):
        image = image.convert('RGBA')
        bg = Image.new("RGB", image.size, (255, 255, 255))
        bg.paste(image, mask=image.split()[3])
        return bg
    return image.convert('RGB')

# --- Трансформации ---
def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(15),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.1)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])


# --- Загрузка данных ---
def load_datasets(train_dir, val_dir, check_dir):
    train_dataset = ImageFolder(train_dir, transform=get_transforms(True), loader=safe_pil_loader)
    val_dataset = ImageFolder(val_dir, transform=get_transforms(False), loader=safe_pil_loader)
    check_dataset = ImageFolder(check_dir, transform=get_transforms(False), loader=safe_pil_loader)
    return train_dataset, val_dataset, check_dataset