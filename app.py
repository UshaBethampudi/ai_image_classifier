# app.py (improved & more robust)
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import gradio as gr
import numpy as np
import cv2
import traceback

MODEL_PATH = "models/best_model.pth"
CLASS_INDEX_PATH = "models/class_index.json"
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

    # Get class names from checkpoint if present, else JSON file
    if isinstance(ckpt, dict) and "classes" in ckpt:
        classes = ckpt["classes"]
    elif os.path.exists(CLASS_INDEX_PATH):
        with open(CLASS_INDEX_PATH, "r") as f:
            class_dict = json.load(f)
        # class_dict might be {"0": "cat", "1": "dog"} -> we want list sorted by keys
        try:
            # attempt to preserve numeric ordering if keys are numeric strings
            classes = [class_dict[k] for k in sorted(class_dict, key=lambda x: int(x))]
        except Exception:
            classes = list(class_dict.values())
    else:
        raise ValueError("No class names found in checkpoint or JSON.")

    # If ckpt is a full dict with model_state_dict, use that; otherwise treat ckpt as state_dict
    state_dict = ckpt.get("model_state_dict", None) if isinstance(ckpt, dict) else None
    if state_dict is None:
        # ckpt might itself be a state_dict (mapping param_name -> tensor)
        state_dict = ckpt if isinstance(ckpt, dict) else None

    model = build_model(len(classes))
    if state_dict is None:
        raise ValueError("No model weights found in the checkpoint.")
    # load with strict=False to be tolerant of minor naming differences
    model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE).eval()
    return model, classes

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Train the model first. Missing best_model.pth.")

model, CLASS_NAMES = load_model(MODEL_PATH)

# Utility: get last Conv2d module to use as Grad-CAM target
def find_last_conv_module(model):
    last_conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    if last_conv is None:
        # fallback: try to use the last item of model.features if it's not conv
        if hasattr(model, "features") and len(list(model.features)) > 0:
            return list(model.features)[-1]
        raise RuntimeError("Could not find a Conv2d layer for Grad-CAM.")
    return last_conv

# Grad-CAM
class GradCAM:
    def __init__(self, model, target_module):
        self.model = model
        self.target_module = target_module
        self.activations = None
        self.gradients = None
        # register a forward hook that also attaches a gradient hook to the output tensor
        target_module.register_forward_hook(self._save_activation_and_attach_grad_hook)

    def _save_activation_and_attach_grad_hook(self, module, input, output):
        # Save the forward activations
        self.activations = output.detach()
        # Attach a hook to the activation's gradient so we can save it during backward()
        def _grad_hook(grad):
            # grad is the gradient of the loss w.r.t. the activation output
            self.gradients = grad.detach()
        # output might be a tensor or tuple; handle tensor case
        try:
            output.register_hook(_grad_hook)
        except Exception:
            # if output isn't a tensor we gracefully ignore (unlikely for conv outputs)
            pass

    def generate(self):
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Activations or gradients not captured for Grad-CAM")
        # activations: [B, C, H, W], gradients: [B, C, H, W]
        acts = self.activations[0]        # [C, H, W]
        grads = self.gradients[0]         # [C, H, W]
        weights = grads.mean(dim=(1, 2))  # [C]
        cam = torch.sum(weights[:, None, None] * acts, dim=0)
        cam = F.relu(cam)
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam.cpu().numpy()

# pick the last conv module dynamically
try:
    target_conv = find_last_conv_module(model)
except Exception as e:
    # If we can't find one, keep target_conv as None and Grad-CAM will error clearly
    target_conv = None

gradcam = GradCAM(model, target_conv) if target_conv is not None else None

def predict(img: Image.Image):
    # Return types must match Gradio outputs: Label (dict) and Image (PIL)
    if img is None:
        return {"Error": 1.0}, None

    try:
        img = img.convert("RGB")
    except Exception:
        return {"Error: Invalid image": 1.0}, None

    try:
        x = transform(img).unsqueeze(0).to(DEVICE)  # [1, C, H, W]

        # Ensure grads are enabled when we need them for Grad-CAM
        with torch.enable_grad():
            x.requires_grad_(True)  # ensure input has grad enabled (not strictly necessary for CAM but safe)
            logits = model(x)  # forward
            probs = torch.softmax(logits, dim=1)[0]
            class_idx = int(torch.argmax(probs).item())

            # Grad-CAM: zero grads, create one-hot, backward
            model.zero_grad(set_to_none=True)
            one_hot = torch.zeros_like(logits).to(DEVICE)
            one_hot[0, class_idx] = 1.0
            logits.backward(gradient=one_hot)

            if gradcam is None:
                # if no gradcam configured, return prediction only
                conf = {CLASS_NAMES[i]: float(probs[i].item()) for i in range(len(CLASS_NAMES))}
                return conf, None

            cam = gradcam.generate()  # returns HxW numpy float in [0,1]
        
        # Post-process CAM -> colored overlay
        cam_resized = cv2.resize(cam, (224, 224))
        heatmap = np.uint8(255 * cam_resized)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        img_resized = cv2.cvtColor(np.array(img.resize((224, 224))), cv2.COLOR_RGB2BGR)
        overlay = cv2.addWeighted(img_resized, 0.6, heatmap, 0.4, 0)
        overlay_pil = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))

        conf = {CLASS_NAMES[i]: float(probs[i].item()) for i in range(len(CLASS_NAMES))}
        return conf, overlay_pil

    except Exception as e:
        # print full traceback to console (very helpful when debugging Gradio errors)
        print("Error in predict():", e)
        traceback.print_exc()
        return {"Error": 1.0}, None

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## Image Classifier with Confidence + Grad-CAM")
    with gr.Row():
        inp = gr.Image(type="pil", label="Upload / Webcam", sources=["upload", "webcam"])
        out1 = gr.Label(num_top_classes=1, label="Prediction")
        out2 = gr.Image(label="Grad-CAM Heatmap")
    # use submit instead of change to avoid repeated triggers while interacting with webcam widget
    inp.change(predict, inputs=inp, outputs=[out1, out2])

if __name__ == "__main__":
    demo.launch()
