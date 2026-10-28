"""
=============================================================================
Project 06: Softmax Clothing Classifier on FashionMNIST in PyTorch
=============================================================================
Author     : Gideon Mwiti
Challenge  : 100 PyTorch Projects in 10 Days
Topic      : Multi-Class Classification, Softmax Probability Layer, Cross-Entropy
Goal       : Classify 28x28 grayscale clothing items into 10 categories
=============================================================================
"""

import os
import sys
import types

# Ensure safe import of torchvision under all environments
sys.modules['bz2'] = types.SimpleNamespace(open=open)
sys.modules['lzma'] = types.SimpleNamespace(open=open)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

class FashionMNISTSoftmaxClassifier(nn.Module):
    """
    Deep Feedforward Classifier with an explicit Softmax output layer:
    - Flatten: 28x28 -> 784
    - Layer 1: 784 -> 512 (BatchNorm, LeakyReLU, Dropout)
    - Layer 2: 512 -> 256 (BatchNorm, LeakyReLU, Dropout)
    - Layer 3: 256 -> 128 (BatchNorm, LeakyReLU)
    - Output: 128 -> 10 Logits
    """
    def __init__(self, input_dim: int = 784, hidden1: int = 512, hidden2: int = 256, hidden3: int = 128, num_classes: int = 10, dropout_rate: float = 0.2):
        super(FashionMNISTSoftmaxClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden2, hidden3),
            nn.BatchNorm1d(hidden3),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Linear(hidden3, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Returns unnormalized logits for CrossEntropyLoss
        return self.net(x)

    def predict_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        # Applies explicit Softmax activation: P(y = k | x) = exp(z_k) / sum(exp(z_j))
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)

def get_dataloaders(data_dir: str, batch_size: int = 64):
    """
    Creates FashionMNIST DataLoaders with standard z-score normalization
    (FashionMNIST mean=0.2860, std=0.3530).
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])

    train_dataset = datasets.FashionMNIST(root=data_dir, train=True, download=False, transform=transform)
    test_dataset = datasets.FashionMNIST(root=data_dir, train=False, download=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    return train_loader, test_loader, test_dataset

def train_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, device: torch.device):
    """Executes one training epoch returning average loss and accuracy."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(logits, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, (correct / total) * 100.0

def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device):
    """Evaluates the model on test data, computing Top-1/Top-3 accuracy and confusion matrix."""
    model.eval()
    running_loss = 0.0
    correct = 0
    top3_correct = 0
    total = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)

            running_loss += loss.item() * images.size(0)
            
            # Top-1 Accuracy
            _, preds = torch.max(logits, 1)
            correct += (preds == labels).sum().item()
            
            # Top-3 Accuracy
            _, top3_preds = torch.topk(logits, k=3, dim=1)
            top3_correct += sum([labels[i].item() in top3_preds[i] for i in range(labels.size(0))])
            
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    test_loss = running_loss / total
    test_acc = (correct / total) * 100.0
    top3_acc = (top3_correct / total) * 100.0

    # Build 10x10 Confusion Matrix
    cm = np.zeros((10, 10), dtype=int)
    for t, p in zip(all_targets, all_preds):
        cm[t, p] += 1

    return test_loss, test_acc, top3_acc, cm

def plot_comprehensive_results(
    history: dict,
    cm: np.ndarray,
    test_dataset,
    model: nn.Module,
    device: torch.device,
    output_path: str
):
    """
    Renders a 4-panel diagnostic report:
    1. Sample test predictions with Softmax confidence levels across categories
    2. Training vs Validation Cross-Entropy Loss Convergence
    3. Training vs Validation Accuracy Progression (%)
    4. 10x10 Confusion Matrix Heatmap annotated with category names
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "Project 06: FashionMNIST Clothing Classifier (PyTorch Softmax Architecture)\n"
        f"Test Accuracy = {history['test_acc'][-1]:.2f}% | Top-3 Accuracy = {history['top3_acc'][-1]:.2f}% | Test Loss = {history['test_loss'][-1]:.4f}",
        fontsize=14,
        fontweight="bold",
        y=0.98
    )

    # 1. Sample Predictions (10 clothing items)
    ax1 = axes[0, 0]
    ax1.axis("off")
    ax1.set_title("1. Model Softmax Inferences on Test Apparel Samples", fontweight="bold", fontsize=11, pad=12)

    sample_subfig = ax1.inset_axes([0.0, 0.0, 1.0, 1.0])
    sample_subfig.axis("off")

    sample_indices = [0, 1, 2, 3, 4, 6, 8, 12, 19, 23]

    for idx_pos, img_idx in enumerate(sample_indices):
        raw_img, true_label = test_dataset[img_idx]
        with torch.no_grad():
            model.eval()
            probs = model.predict_probabilities(raw_img.unsqueeze(0).to(device)).squeeze().cpu().numpy()
            pred_label = int(np.argmax(probs))
            conf = probs[pred_label] * 100.0

        row = idx_pos // 5
        col = idx_pos % 5
        sub_ax = sample_subfig.inset_axes([col * 0.20 + 0.01, (1 - row) * 0.48, 0.17, 0.44])

        # De-normalize image
        disp_img = raw_img.squeeze().numpy() * 0.3530 + 0.2860
        sub_ax.imshow(disp_img, cmap="inferno")
        sub_ax.axis("off")
        color = "#059669" if pred_label == true_label else "#DC2626"
        sub_ax.set_title(
            f"True: {CLASS_NAMES[true_label]}\nPred: {CLASS_NAMES[pred_label]}\n({conf:.1f}%)",
            fontsize=7.5,
            color=color,
            fontweight="bold"
        )

    # 2. Cross-Entropy Loss Curves
    ax2 = axes[0, 1]
    epochs_range = range(1, len(history["train_loss"]) + 1)
    ax2.plot(epochs_range, history["train_loss"], color="#2563EB", linewidth=2.2, label="Train Loss", marker="o")
    ax2.plot(epochs_range, history["test_loss"], color="#EF4444", linewidth=2.2, label="Test Loss", marker="s")
    ax2.set_title("2. Multi-Class Cross-Entropy Loss Convergence", fontweight="bold", fontsize=11)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Cross-Entropy Loss")
    ax2.legend(frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.5)

    # 3. Accuracy Progression Curves
    ax3 = axes[1, 0]
    ax3.plot(epochs_range, history["train_acc"], color="#2563EB", linewidth=2.2, label="Train Top-1 Acc (%)", marker="o")
    ax3.plot(epochs_range, history["test_acc"], color="#10B981", linewidth=2.2, label="Test Top-1 Acc (%)", marker="s")
    ax3.plot(epochs_range, history["top3_acc"], color="#8B5CF6", linestyle="--", linewidth=2.0, label="Test Top-3 Acc (%)")
    ax3.set_title("3. Multi-Class Classification Accuracy (%)", fontweight="bold", fontsize=11)
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Accuracy (%)")
    ax3.set_ylim([75, 100])
    ax3.legend(loc="lower right", frameon=True)
    ax3.grid(True, linestyle="--", alpha=0.5)

    # 4. 10x10 Confusion Matrix Heatmap
    ax4 = axes[1, 1]
    im = ax4.imshow(cm, interpolation="nearest", cmap="Blues")
    ax4.set_title("4. Fashion Category Confusion Matrix (10 Classes)", fontweight="bold", fontsize=11)
    plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
    tick_marks = np.arange(10)
    ax4.set_xticks(tick_marks)
    ax4.set_yticks(tick_marks)
    ax4.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=8)
    ax4.set_yticklabels(CLASS_NAMES, fontsize=8)
    ax4.set_xlabel("Predicted Clothing Category", fontsize=10)
    ax4.set_ylabel("True Ground Truth Label", fontsize=10)

    thresh = cm.max() / 2.0
    for i in range(10):
        for j in range(10):
            val = cm[i, j]
            ax4.text(j, i, f"{val}", ha="center", va="center",
                     color="white" if val > thresh else "black",
                     fontsize=6.5 if val > 999 else 7.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Visual report saved successfully at: {output_path}")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")
    output_plot_path = os.path.join(script_dir, "fashion_mnist_classification_results.png")

    BATCH_SIZE = 64
    EPOCHS = 6
    LEARNING_RATE = 0.001
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=================================================================")
    print("      PROJECT 06: FASHION-MNIST SOFTMAX CLASSIFIER IN PYTORCH    ")
    print("=================================================================")
    print(f"Execution Device    : {DEVICE}")
    print(f"Dataset             : FashionMNIST (60,000 Train, 10,000 Test Grayscale 28x28)")
    print(f"Classes (10 Items)  : {', '.join(CLASS_NAMES)}")
    print(f"Hyperparameters     : Epochs = {EPOCHS}, Batch Size = {BATCH_SIZE}, LR = {LEARNING_RATE}")
    print("-----------------------------------------------------------------")

    # Step 1: Load Data
    train_loader, test_loader, test_dataset = get_dataloaders(data_dir=data_dir, batch_size=BATCH_SIZE)

    # Step 2: Initialize Architecture
    model = FashionMNISTSoftmaxClassifier(
        input_dim=784,
        hidden1=512,
        hidden2=256,
        hidden3=128,
        num_classes=10,
        dropout_rate=0.2
    ).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    history = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
        "top3_acc": []
    }

    print(f"{'Epoch':^7} | {'Train Loss':^12} | {'Train Acc (%)':^14} | {'Test Loss':^12} | {'Test Acc (%)':^14} | {'Top-3 Acc (%)':^14}")
    print("-----------------------------------------------------------------")

    # Step 3: Training Loop
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        test_loss, test_acc, top3_acc, cm = evaluate(model, test_loader, criterion, DEVICE)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        history["top3_acc"].append(top3_acc)

        print(f"{epoch:^7d} | {train_loss:^12.4f} | {train_acc:^14.2f} | {test_loss:^12.4f} | {test_acc:^14.2f} | {top3_acc:^14.2f}")

    print("-----------------------------------------------------------------")
    print(f"Training Complete! Top-1 Accuracy: {test_acc:.2f}% | Top-3 Accuracy: {top3_acc:.2f}% | Test Loss: {test_loss:.4f}")
    print("=================================================================\n")

    # Step 4: Render Visual Artifact
    plot_comprehensive_results(
        history=history,
        cm=cm,
        test_dataset=test_dataset,
        model=model,
        device=DEVICE,
        output_path=output_plot_path
    )

if __name__ == "__main__":
    main()

