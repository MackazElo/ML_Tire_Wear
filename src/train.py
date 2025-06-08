import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from data_loader import get_data_loaders
from model import get_model

def train_model(data_path, num_epochs=10, batch_size=32, lr=0.001, save_path="model_best.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaders = get_data_loaders(data_path, batch_size=batch_size, augment=True)
    train_loader = loaders.get("train")
    val_loader = loaders.get("val")

    model = get_model(num_classes=2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        val_acc = evaluate(model, val_loader, device)

        print(f"Epoch {epoch+1}: Train Loss={total_loss:.4f}, Train Acc={train_acc:.2f}, Val Acc={val_acc:.2f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"✅ Zapisano nowy najlepszy model (Val Acc={val_acc:.2f}) do {save_path}")

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0.0

if __name__ == "__main__":
    train_model(data_path="data", num_epochs=15)
