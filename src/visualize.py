import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from data_loader import get_data_loaders
from model import get_model

def plot_confusion(model, loader, class_names, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues', values_format='d')
    plt.title("Confusion Matrix")
    plt.show()

def show_predictions(model, loader, class_names, device, n=6):
    model.eval()
    images_shown = 0

    fig, axes = plt.subplots(1, n, figsize=(n * 3, 3))
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            for i in range(len(images)):
                if images_shown >= n:
                    break

                img = images[i].cpu().permute(1, 2, 0)  # CxHxW -> HxWxC
                img = img * 0.5 + 0.5  # od-norm
                img = img.numpy()

                color = "green" if preds[i] == labels[i].to(device) else "red"
                axes[images_shown].imshow(img)
                axes[images_shown].set_title(f"True: {class_names[labels[i]]}\nPred: {class_names[preds[i]]}", color=color)
                axes[images_shown].axis("off")

                images_shown += 1

            if images_shown >= n:
                break

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_path = "../data"
    class_names = ["new", "worn"]

    loaders = get_data_loaders(data_path, batch_size=8)
    model = get_model(num_classes=len(class_names))
    model.load_state_dict(torch.load("model_best.pt", map_location=device))
    model = model.to(device)

    print("📊 Macierz pomyłek:")
    plot_confusion(model, loaders["val"], class_names, device)

    print("\n🔍 Przykładowe predykcje:")
    show_predictions(model, loaders["val"], class_names, device, n=6)
