# 🔬 AI for Tele-Dermatology
### Skin Disease Classification & Lesion Segmentation
**Course:** Applied Computer Vision | B.Tech III Year — Woxsen University  
**Submitted by:** Sampriti Mohanty (23WU0102175)  
**Guided by:** Dr. Ravi Kiran Kummamuru

---

## 📌 Overview

This project builds and evaluates deep learning models for two core tele-dermatology tasks:

- **Classification** — Multi-class skin disease recognition using ResNet50 (7 disease classes)
- **Segmentation** — Pixel-level lesion boundary detection using U-Net with ResNet34 encoder

A key focus of this study is evaluating model **robustness** under real-world conditions — specifically brightness variation and skin-tone diversity — and analyzing their impact on both tasks.

---

## 📂 Dataset

**ISIC 2018 Challenge Dataset** (publicly available)

| Task | Samples | Purpose |
|------|---------|---------|
| Task 3 | 10,015 images | Multi-class classification (7 disease types) |
| Task 1 | 2,594 image-mask pairs | Lesion segmentation |

**Disease Classes:** MEL, NV, BCC, AKIEC, BKL, DF, VASC

> ⚠️ The dataset has significant class imbalance — NV accounts for 66.9% of samples.

---

## 🏗️ Models

### Classification — ResNet50
- Pretrained on ImageNet, fine-tuned for 7-class skin disease classification
- Input: 224×224 RGB images
- Optimizer: Adam (lr=1e-4), StepLR scheduler
- Loss: Cross-Entropy
- Trained for 15 epochs on Google Colab (Tesla T4 GPU)

### Segmentation — U-Net (ResNet34 encoder)
- Built using `segmentation-models-pytorch`
- Input: 256×256 RGB images
- Loss: Dice Loss + Binary Cross-Entropy
- Trained for 10 epochs, with and without brightness augmentation

---

## 📊 Results

### Classification (ResNet50)

| Metric | Score |
|--------|-------|
| Accuracy | 87.09% |
| Weighted Precision | 86.74% |
| Weighted Recall | 87.09% |
| Weighted F1-Score | 86.72% |

> ⚠️ MEL (Melanoma) has the lowest recall (0.57) — clinically the most critical failure.

### Segmentation (U-Net)

| Configuration | Dice Score | IoU |
|---------------|------------|-----|
| With Brightness Augmentation | **0.8901** | **0.8045** |
| Without Brightness Augmentation | 0.8709 | 0.7765 |

---

## 🌟 Key Findings

- **Brightness variation** causes up to **19.5% accuracy drop** in classification at extreme illumination (factor 2.0)
- **Segmentation is more robust** to brightness shifts than classification — pixel-level spatial features are less disrupted by uniform illumination changes
- **Skin-tone bias** is significant: dark-toned images achieve only 66.67% accuracy vs. 87.82% for light-toned images (gap of ~21 percentage points), driven by severe underrepresentation in the ISIC dataset
- **Missed lesions** (6% of segmentation test samples) are the dominant segmentation error type, particularly for small or low-contrast lesions
- **Brightness augmentation during training** improves segmentation Dice by +2.2% and IoU by +3.6%

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.x |
| Deep Learning | PyTorch, torchvision |
| Segmentation | segmentation-models-pytorch (SMP) |
| Augmentation | Albumentations |
| Image Processing | OpenCV, PIL |
| Evaluation | scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Platform | Google Colab (NVIDIA Tesla T4 GPU) |

---

## 🚀 How to Run

1. Open `ACV_Case_Study.ipynb` in [Google Colab](https://colab.research.google.com/)
2. Run the dataset download cells — images are fetched directly from the ISIC S3 bucket
3. Follow the notebook sections in order:
   - Data preprocessing & EDA
   - Classification model training & evaluation
   - Brightness robustness analysis
   - Segmentation model training & evaluation
   - Error analysis & visualizations

> 💡 A GPU runtime is strongly recommended (Runtime → Change runtime type → T4 GPU)

---

## 📁 Repository Structure

```
├── ACV_Case_Study.ipynb          # Main notebook (classification + segmentation)
├── ACV_Case_Study_Report.pdf     # Full case study report
└── README.md                     # This file
```

---

## 📚 References

- Esteva et al. (2017) — Dermatologist-level classification of skin cancer with deep neural networks
- Ronneberger et al. (2015) — U-Net: Convolutional Networks for Biomedical Image Segmentation
- Zhou et al. (2018) — UNet++: A Nested U-Net Architecture
- Groh et al. (2021) — Evaluating deep neural networks trained on clinical images in dermatology
- [ISIC 2018 Challenge](https://challenge2018.isic-archive.com/)
