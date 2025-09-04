# train.py
import os, time, json, argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import datasets, models
from utils import build_transforms, save_class_index

def train(data_dir, epochs, batch_size, lr, device):
    train_tf = build_transforms(train=True)
    val_tf = build_transforms(train=False)

    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_tf)
    val_ds = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    num_classes = len(train_ds.classes)
    print("Classes:", train_ds.classes)

    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V2)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)

    best_acc = 0.0
    os.makedirs("models", exist_ok=True)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)

        # validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
        val_acc = correct / total
        val_loss = running_loss / len(train_ds)

        scheduler.step(val_loss)
        print(f"[{epoch+1}/{epochs}] loss: {running_loss/len(train_ds):.4f} val_acc: {val_acc:.4f}")

        # save best
        if val_acc > best_acc:
            best_acc = val_acc
            ckpt = {
                "model_state_dict": model.state_dict(),
                "classes": train_ds.classes,
                "model_name": "mobilenet_v2"
            }
            torch.save(ckpt, os.path.join("models", "best_model.pth"))
            print("Saved best model (val_acc)", best_acc)

    # save classes json (again)
    save_class_index(train_ds.classes, path="models/class_index.json")
    print("Training complete. Best val acc:", best_acc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    train(args.data, args.epochs, args.batch_size, args.lr, args.device)
