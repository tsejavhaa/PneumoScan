import torch
import torch.nn as nn
from torchvision import models

def get_model(num_classes=2, pretrained=True):
    model = models.efficientnet_b0(pretrained=pretrained)
    # EfficientNet-B0-ийн classifier-г сольж байна
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.classifier[1].in_features, num_classes)
    )
    return model