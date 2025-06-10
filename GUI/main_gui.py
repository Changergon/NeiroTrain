import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, Label, Button

from models.improved_cnn import ImprovedCNN  # твоя модель
import os

# === Настройки ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 9
model = ImprovedCNN(
    num_classes=num_classes,
    use_cbam=True,
    cbam_ratio=16,
    cbam_kernel_size=9,
    use_dropblock=True,
    drop_prob=0.1,
    block_size=5,
    use_batchnorm=True
)
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "best_model.pth"))
model = torch.load(model_path, map_location=device, weights_only=False)
model.eval()

idx_to_class = {
    0: "BABY_PRODUCTS",
    1: "BEAUTY_HEALTH",
    2: "CLOTHING_ACCESSORIES_JEWELLERY",
    3: "ELECTRONICS",
    4: "GROCERY",
    5: "HOBBY_ARTS_STATIONERY",
    6: "HOME_KITCHEN_TOOLS",
    7: "PET_SUPPLIES",
    8: "SPORTS_OUTDOOR"
}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# === GUI интерфейс ===
class ImageClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Классификатор товаров")
        self.image_label = Label(root)
        self.image_label.pack(pady=10)

        self.result_label = Label(root, text="", font=("Arial", 14))
        self.result_label.pack(pady=10)

        self.button = Button(root, text="Загрузить изображение", command=self.load_and_predict)
        self.button.pack(pady=5)

    def load_and_predict(self):
        file_path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.png *.jpeg")])
        if not file_path:
            return

        image = Image.open(file_path).convert("RGB")
        display_image = image.resize((224, 224))
        tk_image = ImageTk.PhotoImage(display_image)
        self.image_label.config(image=tk_image)
        self.image_label.image = tk_image

        tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(tensor)
            probs = F.softmax(output, dim=1)
            top_prob, top_class = torch.max(probs, 1)

        pred_class = idx_to_class[top_class.item()]
        confidence = top_prob.item() * 100
        self.result_label.config(text=f"🧠 Класс: {pred_class}\n📊 Уверенность: {confidence:.2f}%")

# === Запуск приложения ===
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageClassifierApp(root)
    root.mainloop()
