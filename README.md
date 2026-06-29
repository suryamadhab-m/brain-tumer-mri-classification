# Brain Tumor MRI Classification using Lightweight CNN with Grad-CAM Explainability

A lightweight, CPU-deployable Convolutional Neural Network for automated brain tumor detection and classification from MRI images, with built-in Grad-CAM explainability.

---

## Project Overview

Brain tumor diagnosis from MRI scans traditionally depends on expert neuro-radiologists — a resource that is unevenly distributed, especially in smaller hospitals and rural areas in India. Most existing deep learning solutions for this task require GPU infrastructure and offer no visual explanation for their predictions, limiting real-world clinical adoption.

This project addresses both problems:
- A **lightweight CNN** (only 187,364 parameters / 731 KB) that runs on a standard CPU
- **Grad-CAM** integrated directly into the inference pipeline, generating a visual heatmap with every prediction

---

## Demo

![Grad-CAM Heatmaps](gradcam_samples.png)
*Top row: Original MRI scans | Bottom row: Grad-CAM heatmap overlays showing regions the model used to make its prediction*

![Training Curves](training_curves.png)

---

## Results

| Metric | Value |
|--------|-------|
| Overall Accuracy | **83.50%** |
| Macro F1-Score | **0.8268** |
| Total Parameters | **187,364** (731 KB) |
| CPU Inference Time | **139.03 ms / image** |
| GPU Required? | **No** |

### Per-Class Performance

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Glioma | 1.0000 | 0.5850 | 0.7382 |
| Meningioma | 0.7731 | 0.7750 | 0.7740 |
| No Tumor | 0.8305 | 0.9925 | 0.9043 |
| Pituitary | 0.8111 | 0.9875 | 0.8906 |

![Confusion Matrix](confusion_matrix.png)

---

## Model Architecture

The model uses depthwise-separable convolutions to keep parameter count minimal:

```
Input (150x150x3)
→ Conv2D (32 filters) + BatchNorm + MaxPool
→ SeparableConv2D (64 filters) + BatchNorm + MaxPool + Dropout
→ SeparableConv2D (128 filters) + BatchNorm + MaxPool + Dropout
→ SeparableConv2D (128 filters) + BatchNorm + MaxPool + Dropout
→ Conv2D (128 filters) [last_conv — used for Grad-CAM]
→ GlobalAveragePooling
→ Dense (64) + Dropout
→ Dense (4) + Softmax
```

**Total parameters: 187,364 (731 KB)**
Compare: VGG16 = 138M params, ResNet50 = 25M params

---

## Dataset

**Brain Tumor MRI Dataset** by Masoud Nickparvar (Kaggle)

- 4 classes: Glioma, Meningioma, Pituitary Tumor, No Tumor
- Training: 5,600 images (split 85/15 train/val)
- Test: 1,600 images (400 per class)

Download: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

---

## How to Run

### Option 1: Google Colab (Recommended)
1. Open the `.ipynb` file in Google Colab
2. Set Runtime → Change runtime type → **T4 GPU**
3. Run cells top to bottom
4. Upload your `kaggle.json` when prompted (Cell 2)

### Option 2: Local Setup
```bash
pip install tensorflow scikit-learn matplotlib grad-cam kaggle
jupyter notebook Brain_Tumor_MRI_Classification_Colab_Suryamadhab_Moharana.ipynb
```

---

## Key Features

- **Lightweight** — 187,364 parameters, runs on CPU without GPU
- **Explainable** — Grad-CAM heatmap generated for every prediction automatically
- **Fast** — 139 ms average CPU inference time per image
- **Practical** — Designed for deployment in resource-constrained clinical settings

---

## Tech Stack

- Python 3.12
- TensorFlow 2.20
- Grad-CAM
- scikit-learn
- Matplotlib
- Google Colab (T4 GPU for training)

---

## Project Context

This project was developed as part of a research internship under **Dr. Satya Ranjan Dash**, Professor & Dean, School of Computer Applications, KIIT Deemed to be University, Bhubaneswar, in the domain of Biomedical Image Processing, Computer Vision, and Medical Image Analysis.

---

## Author

**Suryamadhab Moharana**
B.Tech CSE, ITER, SOA University, Bhubaneswar
GitHub: [suryamadhab-m](https://github.com/suryamadhab-m)

---

## License

This project is for academic and research purposes only.
