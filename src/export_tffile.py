# src/export_tflite.py
import torch
import tensorflow as tf
from model import get_model

# PyTorch загвараа ачадна
model = get_model()
model.load_state_dict(torch.load("models/best_pneumonia_model.pth", map_location="cpu"))
model.eval()

# Dummy input
dummy_input = torch.randn(1, 3, 224, 224)

# TorchScript болгох
traced_script_module = torch.jit.trace(model, dummy_input)
traced_script_module.save("models/pneumonia_jit.pt")

# TFLite болгох
converter = tf.lite.TFLiteConverter.from_saved_model("models/pneumonia_jit.pt")  # эсвэл from_pt
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open("app/tflite_model/pneumonia_model.tflite", "wb") as f:
    f.write(tflite_model)

print("TFLite model saved – та Android/iOS-д шууд хийж болно!")