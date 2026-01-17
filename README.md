# PneumoScan: Multi-Task Pneumonia Detection from Chest X-Rays

🩻 **Ex-Surgeon Built AI Tool** for detecting and localizing pneumonia in chest radiographs using deep learning.

![Banner Image](./PneumoScan/data/screeshot.png)  
*(Replace with your actual banner or Gradio demo screenshot)*

## Overview

This project upgrades the classic Pneumonia classification to a **multi-task model**:
- **Classification**: Detects presence of pneumonia (binary: Normal vs Pneumonia)
- **Localization/Segmentation**: Generates approximate bounding box-based mask to highlight opacity regions

Built on the **RSNA Pneumonia Detection Challenge** dataset (~30,000 DICOM images), using **PyTorch**, **Swin Transformer backbone**, and **U-Net head** for segmentation.

As a former **surgeon** with 10+ years in mobile development (Android/iOS) and self-taught ML engineer (TensorFlow Certified), I designed this to bridge clinical knowledge with deployable AI.

**Key Highlights**
- Handles real medical DICOM format with proper windowing (lung window)
- Pseudo-mask generation from provided bounding boxes
- Multi-task learning (weighted classification + Dice segmentation loss)
- Mobile-ready export potential (TFLite compatible)
- Gradio demo for interactive testing

## Features

- DICOM image reading & preprocessing (pydicom + windowing)
- Pseudo-mask creation from competition bounding boxes
- Multi-task model: SwinV2-Tiny backbone + Classification head + U-Net segmentation
- Training with mixed loss (BCE + Dice)
- Evaluation: Accuracy, IoU, ROC-AUC
- Interactive Gradio web demo (classification prob + overlay mask)

## Tech Stack

- **Framework**: PyTorch 2.0+
- **Model**: timm (SwinV2-Tiny) + segmentation_models_pytorch (U-Net)
- **Data Handling**: pydicom, OpenCV, Albumentations/PIL
- **Training**: AdamW, CosineAnnealingLR
- **Demo**: Gradio
- **Other**: numpy, pandas, tqdm, matplotlib, scikit-learn

## Dataset

- **Source**: [RSNA Pneumonia Detection Challenge (Kaggle)](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge)
- **Format**: DICOM (.dcm) images + CSV labels with bounding boxes
- **Classes**: Normal (0) vs Pneumonia (1)
- **Size**: ~26,684 training images (after grouping by patientId)

## Results

(Replace with your actual metrics after training)

- Validation Accuracy: ~92-95%
- Mean IoU (Segmentation): ~0.45-0.60 (depending on pseudo-mask quality)
- Training time: ~30-60 min per 10 epochs on Kaggle P100 GPU

## Demo

Try it live!  
[![Open In Hugging Face](https://img.shields.io/badge/Gradio%20Demo-Open%20in%20Hugging%20Face-blue?logo=huggingface)](https://huggingface.co/spaces/YOUR_USERNAME/pneumoscan-demo)  
*(Upload your trained model to HF Spaces or use local Gradio)*

Example output:
- Input X-ray → Pneumonia Probability: 0.87
- Overlay: Red mask on opacity regions

