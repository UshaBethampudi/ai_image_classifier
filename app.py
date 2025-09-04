import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import gradio as gr
import numpy as np
import cv2

MODEL_PATH = "models/best_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def build_model(num_classes):
    model = models.mobilenet_v2(weights=None)
    in_feat = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_feat, num_classes)
    return model


def load_model(path):
    ckpt = torch.load(path, map_location=DEVICE)
    if "class_names" in ckpt:
        classes = ckpt["class_names"]
    else:
        classes = ["apple", "banana", "cat", "dog", "glass", "plastic"]

    model = build_model(len(classes))
    model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
    model.to(DEVICE).eval()
    return model, classes

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Train the model first. Missing best_model.pth.")

model, CLASS_NAMES = load_model(MODEL_PATH)


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self.save_activations)
        target_layer.register_full_backward_hook(self.save_gradients)

    def save_activations(self, module, input, output):
        self.activations = output.detach()

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, class_idx):
        acts = self.activations[0]
        grads = self.gradients[0]
        weights = grads.mean(dim=(1, 2))
        cam = torch.sum(weights[:, None, None] * acts, dim=0)
        cam = F.relu(cam)
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam.cpu().numpy()

gradcam = GradCAM(model, model.features[-1])


def predict(img: Image.Image):
    if img is None:  # user uploaded nothing
        return {"Error": 1.0}, None

    try:
        img = img.convert("RGB")
    except Exception as e:
        return {"Error: Invalid image": 1.0}, None

    x = transform(img).unsqueeze(0).to(DEVICE)

    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0]
    class_idx = int(torch.argmax(probs).item())


    model.zero_grad(set_to_none=True)
    one_hot = torch.zeros_like(logits)
    one_hot[0, class_idx] = 1
    logits.backward(gradient=one_hot)

    cam = gradcam.generate(class_idx)
    cam = cv2.resize(cam, (224, 224))
    heatmap = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    img_resized = cv2.cvtColor(np.array(img.resize((224, 224))), cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(img_resized, 0.6, heatmap, 0.4, 0)
    overlay_pil = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))

    conf = {CLASS_NAMES[i]: float(probs[i].item()) for i in range(len(CLASS_NAMES))}
    return conf, overlay_pil



with gr.Blocks() as demo:
    gr.Markdown("## Image Classifier with Confidence + Grad-CAM")
    with gr.Row():
        inp = gr.Image(type="pil", label="Upload / Webcam", sources=["upload", "webcam"])
        out1 = gr.Label(num_top_classes=1, label="Prediction")
        out2 = gr.Image(label="Grad-CAM Heatmap")
    inp.change(predict, inputs=inp, outputs=[out1, out2])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=True)
