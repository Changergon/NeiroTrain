import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, Label, Button, Frame, Scrollbar, Canvas

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
        self.root.title("🛒 Классификатор товаров")
        self.root.configure(bg="#E3F2FD")  # Светлый фон

        # Экран приветствия перед загрузкой изображений
        self.welcome_frame = Frame(self.root, bg="#ffffff", bd=2, relief="solid")
        self.welcome_frame.pack(padx=20, pady=20, fill="both", expand=True)

        Label(self.welcome_frame, text="🔍 Добро пожаловать!", font=("Arial", 20, "bold"), bg="#ffffff").pack(pady=10)
        Label(self.welcome_frame, text="Выберите изображения товаров для классификации.", font=("Arial", 14),
              bg="#ffffff").pack(pady=5)

        self.button = Button(self.welcome_frame, text="📁 Загрузить изображения", command=self.load_and_predict,
                             font=("Arial", 14), bg="#2196F3", fg="white", relief="flat")
        self.button.pack(pady=10)

        self.button.bind("<Enter>", lambda e: self.button.config(bg="#1976D2"))
        self.button.bind("<Leave>", lambda e: self.button.config(bg="#2196F3"))

        # Контейнер для изображений (скрыт до загрузки)
        self.image_container = Frame(self.root, bg="#ffffff")
        self.image_canvas = Canvas(self.image_container, bg="#ffffff")
        self.image_scrollbar = Scrollbar(self.image_container, orient="vertical", command=self.image_canvas.yview)
        self.image_frame = Frame(self.image_canvas, bg="#ffffff")

        self.image_frame.bind("<Configure>",
                              lambda e: self.image_canvas.configure(scrollregion=self.image_canvas.bbox("all")))
        self.image_canvas.create_window((0, 0), window=self.image_frame, anchor="nw")
        self.image_canvas.configure(yscrollcommand=self.image_scrollbar.set)

        # Кнопка для дополнительной загрузки изображений
        self.additional_button = Button(self.root, text="📁 Загрузить еще изображения", command=self.load_and_predict,
                                        font=("Arial", 14), bg="#4CAF50", fg="white", relief="flat")
        self.additional_button.bind("<Enter>", lambda e: self.additional_button.config(bg="#388E3C"))
        self.additional_button.bind("<Leave>", lambda e: self.additional_button.config(bg="#4CAF50"))
        self.additional_button.pack(pady=10)
        self.additional_button.pack_forget()  # Скрываем до первой загрузки

    def load_and_predict(self):
        file_paths = filedialog.askopenfilenames(filetypes=[("Images", "*.jpg *.png *.jpeg")])
        if not file_paths:
            return

        # Скрыть приветственный экран и показать контейнер изображений (если еще не скрыт)
        if self.welcome_frame.winfo_ismapped():
            self.welcome_frame.pack_forget()
            self.image_container.pack(padx=20, pady=20, fill="both", expand=True)
            self.image_canvas.pack(side="left", fill="both", expand=True)
            self.image_scrollbar.pack(side="right", fill="y")
            self.additional_button.pack(pady=10)  # Показываем кнопку дополнительной загрузки

        for file_path in file_paths:
            image = Image.open(file_path).convert("RGB")
            display_image = image.resize((120, 120))
            tk_image = ImageTk.PhotoImage(display_image)

            frame = Frame(self.image_frame, bg="#FAFAFA", bd=2, relief="groove")
            frame.pack(pady=5, padx=5, fill="x")

            lbl = Label(frame, image=tk_image, text="", font=("Arial", 14), compound="left", bg="#FAFAFA")
            lbl.image = tk_image
            lbl.pack(side="left", padx=10, pady=5)

            tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(tensor)
                probs = F.softmax(output, dim=1)
                top_prob, top_class = torch.max(probs, 1)

            pred_class = idx_to_class[top_class.item()]
            confidence = top_prob.item() * 100
            Label(frame, text=f"{pred_class}\n{confidence:.2f}% уверенности", font=("Arial", 12), bg="#FAFAFA").pack(
                side="left", padx=10, pady=5)


# === Запуск приложения ===
if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("800x500")  # Увеличенный размер окна
    app = ImageClassifierApp(root)
    root.mainloop()
