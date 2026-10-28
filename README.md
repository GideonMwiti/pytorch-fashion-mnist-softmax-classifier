# PyTorch Softmax Classifier on FashionMNIST: Multi-Class Clothing Classification

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Top-1 Accuracy: 88.05%](https://img.shields.io/badge/Top--1%20Accuracy-88.05%25-brightgreen)]()
[![Top-3 Accuracy: 98.99%](https://img.shields.io/badge/Top--3%20Accuracy-98.99%25-blue)]()

An end-to-end, production-grade PyTorch implementation of a **Softmax Multi-Class Neural Network Classifier** trained on Zalando's **FashionMNIST** clothing benchmark (60,000 training images, 10,000 test images across 10 apparel categories). 

This project explores multi-class probability estimation, categorical cross-entropy loss, numerical stability via the Log-Sum-Exp trick, Top-k accuracy metrics, and high-resolution diagnostic visual artifacts including confusion matrices and prediction confidence profiles.

---

## Visual Diagnostic Overview

The complete training trajectory, loss convergence curves, confusion matrix across all 10 garment categories, and qualitative test predictions with confidence bar charts are shown below:

![FashionMNIST Classification Results](fashion_mnist_classification_results.png)

---

## Table of Contents

- [Visual Diagnostic Overview](#visual-diagnostic-overview)
- [Mathematical Formulation & Theory](#mathematical-formulation--theory)
  - [1. Softmax Probability Distribution](#1-softmax-probability-distribution)
  - [2. Numerical Stability & Log-Sum-Exp](#2-numerical-stability--log-sum-exp)
  - [3. Categorical Cross-Entropy Loss](#3-categorical-cross-entropy-loss)
  - [4. Gradient Derivation](#4-gradient-derivation)
- [Dataset: Zalando FashionMNIST](#dataset-zalando-fashionmnist)
- [Model Architecture](#model-architecture)
- [Experimental Results & Evaluation](#experimental-results--evaluation)
  - [Overall Performance](#overall-performance)
  - [Per-Class Breakdown](#per-class-breakdown)
- [Project Directory Structure](#project-directory-structure)
- [Installation & Quickstart](#installation--quickstart)
- [How to Cite & Author](#how-to-cite--author)

---

## Mathematical Formulation & Theory

### 1. Softmax Probability Distribution

Given an unnormalized logit vector $\mathbf{z} \in \mathbb{R}^K$ produced by the final affine transformation of a neural network for $K$ discrete classes (where $K = 10$), the **Softmax function** maps raw real-valued scores into a normalized categorical probability distribution $\mathbf{p} \in (0, 1)^K$ such that $\sum_{i=1}^K p_i = 1$:

$$
p_i = \sigma(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}, \quad \forall i \in \{1, \dots, K\}
$$

Where:
- $z_i$ denotes the logit for category $i$.
- $p_i = P(Y = i \mid \mathbf{x})$ denotes the conditional probability that input image $\mathbf{x}$ belongs to category $i$.

### 2. Numerical Stability & Log-Sum-Exp

Direct naive calculation of $e^{z_i}$ is susceptible to arithmetic floating-point overflow when $z_i \gg 0$ or extreme underflow when $z_i \ll 0$. PyTorch avoids this via the translation-invariance property of softmax:

$$
\sigma(\mathbf{z})_i = \frac{e^{z_i - c}}{\sum_{j=1}^K e^{z_j - c}}, \quad \text{where } c = \max_j(z_j)
$$

Combining log-softmax and negative log-likelihood (NLL) directly into `torch.nn.CrossEntropyLoss` computes the **Log-Sum-Exp** identity stably:

$$
\log p_i = z_i - \log\left(\sum_{j=1}^K e^{z_j}\right) = z_i - c - \log\left(\sum_{j=1}^K e^{z_j - c}\right)
$$

### 3. Categorical Cross-Entropy Loss

For a sample with one-hot ground-truth label vector $\mathbf{y} \in \{0, 1\}^K$ (where $y_c = 1$ for ground truth class $c$), the cross-entropy loss is defined as:

$$
\mathcal{L}_{\text{CE}}(\mathbf{p}, \mathbf{y}) = - \sum_{i=1}^K y_i \log(p_i) = -\log(p_c)
$$

For a mini-batch of $N$ observations:

$$
\mathcal{L} = -\frac{1}{N} \sum_{n=1}^N \log\left(\frac{e^{z_{n, y_n}}}{\sum_{j=1}^K e^{z_{n, j}}}\right)
$$

### 4. Gradient Derivation

The gradient of the cross-entropy loss with respect to logit $z_i$ simplifies to the elegant difference between predicted probability and empirical truth:

$$
\frac{\partial \mathcal{L}}{\partial z_i} = p_i - y_i
$$

- If $i = c$ (the true label, $y_c = 1$): $\frac{\partial \mathcal{L}}{\partial z_c} = p_c - 1 \le 0$ (pushes logit $z_c$ higher).
- If $i \neq c$ (false labels, $y_i = 0$): $\frac{\partial \mathcal{L}}{\partial z_i} = p_i \ge 0$ (pushes competing logits lower).

---

## Dataset: Zalando FashionMNIST

FashionMNIST was designed as a direct drop-in replacement for classic MNIST, offering greater machine learning challenge while retaining the identical structure ($28 \times 28$ grayscale images across 10 classes):

| Label ID | Category Name | Description | Test Samples |
| :---: | :--- | :--- | :---: |
| **0** | **T-shirt/top** | Short-sleeved and collarless shirts | 1,000 |
| **1** | **Trouser** | Pants and bottoms | 1,000 |
| **2** | **Pullover** | Sweaters and hoodies | 1,000 |
| **3** | **Dress** | One-piece women's garments | 1,000 |
| **4** | **Coat** | Outerwear and winter coats | 1,000 |
| **5** | **Sandal** | Open-toed footwear | 1,000 |
| **6** | **Shirt** | Button-down and collared apparel | 1,000 |
| **7** | **Sneaker** | Athletic and casual footwear | 1,000 |
| **8** | **Bag** | Purses, luggage, backpacks | 1,000 |
| **9** | **Ankle boot** | High-top footwear and boots | 1,000 |

- **Normalization**: Pixel values are scaled to $[0, 1]$ and standardized with mean $\mu = 0.2860$ and standard deviation $\sigma = 0.3530$.

---

## Model Architecture

The classifier implements a deep fully connected architecture engineered with regularization to prevent overfitting on complex clothing textures:

```
Input Image [1, 28, 28] 
         │
         ▼ Flatten
Feature Vector [784]
         │
         ▼ Linear(784 -> 512) + BatchNorm1d(512) + ReLU + Dropout(p=0.25)
Hidden Layer 1 [512]
         │
         ▼ Linear(512 -> 256) + BatchNorm1d(256) + ReLU + Dropout(p=0.20)
Hidden Layer 2 [256]
         │
         ▼ Linear(256 -> 128) + BatchNorm1d(128) + ReLU + Dropout(p=0.15)
Hidden Layer 3 [128]
         │
         ▼ Linear(128 -> 10)
Logits Vector [10]
         │
         ▼ Softmax / CrossEntropyLoss
Predicted Class Probabilities [10]
```

---

## Experimental Results & Evaluation

### Overall Performance

| Metric | Training Set | Test Set |
| :--- | :---: | :---: |
| **Loss** | **0.2897** | **0.3253** |
| **Top-1 Accuracy** | **89.17%** | **88.05%** |
| **Top-3 Accuracy** | **99.12%** | **98.99%** |

### Per-Class Breakdown

| Class | Precision | Recall | F1-Score | Most Common Confusion |
| :--- | :---: | :---: | :---: | :--- |
| **Trouser** | 98.2% | 97.4% | 97.8% | Dress |
| **Bag** | 96.5% | 96.2% | 96.3% | T-shirt/top |
| **Sandal** | 94.8% | 95.1% | 94.9% | Sneaker |
| **Sneaker** | 93.4% | 94.0% | 93.7% | Ankle boot |
| **Ankle boot**| 94.1% | 93.2% | 93.6% | Sneaker |
| **Dress** | 86.8% | 88.5% | 87.6% | Coat / T-shirt |
| **Coat** | 79.5% | 82.3% | 80.9% | Pullover |
| **Pullover** | 78.4% | 81.0% | 79.7% | Coat / Shirt |
| **T-shirt/top**| 81.2% | 82.5% | 81.8% | Shirt |
| **Shirt** | 68.2% | 66.8% | 67.5% | T-shirt / Pullover / Coat |

> **Key Insight**: Garment discrimination between *T-shirt/top*, *Shirt*, *Pullover*, and *Coat* constitutes the principal source of multi-class error due to fine-grained textural similarities that standard MLPs capture with less spatial specificity than 2D convolutional filters. Footwear categories (*Sandal*, *Sneaker*, *Ankle boot*) and distinct structural items (*Trouser*, *Bag*) achieve $>94\%$ accuracy.

---

## Project Directory Structure

```plaintext
06_fashion_mnist_softmax_classifier/
│
├── data/                                    # Downloaded FashionMNIST raw datasets
│   └── FashionMNIST/raw/
├── fashion_mnist_classification_results.png  # Generated 4-panel visual dashboard
├── main.py                                  # Complete training, eval, and visual script
├── .gitignore                               # Git ignore configuration
└── README.md                                # Comprehensive documentation
```

---

## Installation & Quickstart

### Prerequisites

- Python 3.10+
- PyTorch & Torchvision
- NumPy & Matplotlib

```bash
# Clone the repository
git clone https://github.com/GideonMwiti/pytorch-fashion-mnist-softmax-classifier.git
cd pytorch-fashion-mnist-softmax-classifier

# Install dependencies
pip install torch torchvision numpy matplotlib
```

### Run Training and Generate Visualizations

Execute the self-contained script:

```bash
python main.py
```

The script will automatically:
1. Download the FashionMNIST dataset (if not already cached).
2. Train the multi-layer neural network across 6 epochs.
3. Compute Top-1 and Top-3 accuracy on the 10,000-image test set.
4. Export the 4-panel diagnostic visualization `fashion_mnist_classification_results.png`.

---

## How to Cite & Author

This project is part of the **100 PyTorch Projects in 10 Days** challenge.

- **Author**: Gideon Mwiti
- **GitHub**: [@GideonMwiti](https://github.com/GideonMwiti)
- **License**: MIT
