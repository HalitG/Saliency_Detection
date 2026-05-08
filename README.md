# Saliency Object Detection (SOD) Pipeline

An end-to-end Machine Learning pipeline for Saliency Object Detection, developed as part of the **Genpact GigaAcademy AI Engineering Internship**. This project features a custom PyTorch U-Net architecture designed to identify and segment the most visually conspicuous objects in an image.

## 🚀 Project Overview
This repository contains a complete deep learning workflow, transitioning from a baseline Encoder-Decoder CNN to an optimized U-Net with skip connections. The pipeline includes data preprocessing, custom hybrid loss implementation, rigorous evaluation, and a real-time web deployment via Gradio.

## 📊 Dataset
The model was trained and evaluated using the **MSRA10K** dataset, which consists of 10,000 images with pixel-level saliency masks.
* **Source:** [MSRA10K Saliency Object Segmentation (Kaggle)](https://www.kaggle.com/datasets/evvalaycan/saliency-object-segmentation-msra10k)

## 🧠 Model Architecture
The final model utilizes a **U-Net** architecture to ensure high-fidelity edge detection:
* **Encoder:** Progressive feature extraction using convolutional layers and max-pooling.
* **Skip Connections:** Direct concatenation of high-resolution encoder maps to the decoder path to prevent spatial information loss.
* **Regularization:** Integrated Batch Normalization and Dropout ($p=0.5$) to ensure robust generalization.
* **Hybrid Loss:** Optimized using a combination of **Binary Cross-Entropy (BCE)** and **Intersection over Union (IoU)**.

## 📈 Performance Results
The improved U-Net achieved the following benchmarks on the MSRA10K test set:
* **Mean IoU (mIoU):** 0.6720
* **F1-Score:** 0.8030
* **Mean Absolute Error (MAE):** 0.0890
* **Inference Latency:** ~182ms per image

## 📂 Repository Structure
```text
├── sod_model.py             # U-Net architecture definition
├── data_loader.py           # Custom Dataset and Dataloader classes
├── train.py                 # Training script with Hybrid Loss
├── evaluate.py              # Performance benchmarking script
├── app.py                   # Gradio web interface for real-time demo
├── SOD_Main_Project.ipynb   # Full development and analysis notebook
├── improved_model.pth       # Final trained weights
├── Project_Report.pdf       # Detailed technical internship report
└── Genpact_SOD_Presentation.pptx # Final project presentation
