# PneumoScan: Pneumonia Detection from Chest X-Rays (v1 – Baseline)

🩻 **Built by an Ex-Surgeon turned ML Engineer**  
A simple yet powerful deep learning model for detecting pneumonia from chest X-ray images.

![Normal vs Pneumonia X-Ray Examples](https://i.ytimg.com/vi/xaz0Uo0SIL4/maxresdefault.jpg)  
*Left: Normal lungs | Right: Pneumonia with consolidation (source: educational medical video)*

### Project Overview

This is the **first version (baseline)** of PneumoScan – a binary classification model that distinguishes **NORMAL** from **PNEUMONIA** in chest X-rays.

- Dataset: Kaggle Chest X-Ray Images (Pneumonia) – ~5,863 images
- Model: Pretrained **EfficientNet-B0** (transfer learning)
- Achieved **Test Accuracy: 94.71%** on hold-out test set
- Precision/Recall/F1 strong for pneumonia class (critical for clinical use)

As a former **surgeon** (with experience in thoracic cases), 10+ years mobile developer (Android/iOS), and self-taught ML engineer (TensorFlow Certified), I built this to create clinically relevant AI tools.

### Key Results

- **Test Accuracy**: 0.9471 (94.71%)
- **Classification Report**:

## Demo

Try it live!  
![Banner Image](./data/screeshot.png)  
[![Open In Hugging Face](https://img.shields.io/badge/Gradio%20Demo-Open%20in%20Hugging%20Face-blue?logo=huggingface)](https://huggingface.co/spaces/tsejavhaa/pneumoscan-pneumonia-xray)  


Here is the confusion matrix from the test set:

![Confusion Matrix Example for EfficientNet Pneumonia Model](https://www.researchgate.net/publication/365951796/figure/fig15/AS:11431281104283681@1669987851976/The-EfficientNet-lite-B0-confusion-matrix-of-the-multi-class-classification-One-can.ppm)  
*Similar pattern to our model: High recall for Pneumonia, low false negatives*

More sample X-ray comparisons:

![Normal vs Pneumonia Samples](https://www.researchgate.net/publication/355191637/figure/fig2/AS:11431281360957476@1744111369078/Difference-in-Chest-X-Ray-Images-in-Normal-and-Pneumonia.png)  
*Clear visual difference: Pneumonia shows opacities/consolidation*

### Tech Stack

- **Framework**: PyTorch 2.0+
- **Model**: torchvision.models.efficientnet_b0 (pretrained on ImageNet)
- **Data**: torchvision.datasets.ImageFolder + transforms
- **Optimizer**: AdamW (lr=3e-4, weight decay=1e-4)
- **Scheduler**: CosineAnnealingLR
- **Evaluation**: sklearn (classification_report, confusion_matrix), seaborn/matplotlib
- **Training**: 12 epochs on GPU (~15-25 min)

### Dataset

- Source: [Kaggle – Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- Structure: train/val/test folders with NORMAL and PNEUMONIA subfolders
- Classes: ['NORMAL', 'PNEUMONIA']
