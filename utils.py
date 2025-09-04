# utils.py
from torchvision import transforms
import json
from PIL import Image
import torch

IMG_SIZE = 224

def build_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.1,0.1,0.1,0.05),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
        ])

def preprocess_pil(img: Image.Image):
    tf = build_transforms(train=False)
    return tf(img).unsqueeze(0)

def save_class_index(classes, path="class_index.json"):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({i: c for i, c in enumerate(classes)}, f, ensure_ascii=False, indent=2)

def load_class_index(path="class_index.json"):
    with open(path, "r", encoding="utf-8") as f:
        m = json.load(f)
    return {int(k): v for k,v in m.items()}
