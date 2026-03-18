"""
Evaluation metrics utilities for PyTorch.
Provides common metrics and visualization tools.
"""

import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class MetricsTracker:
    """
    Track and compute training/validation metrics.

    Examples:
        >>> tracker = MetricsTracker()
        >>> tracker.update(loss=0.5, correct=8, total=10)
        >>> tracker.update(loss=0.3, correct=9, total=10)
        >>> print(f"Average loss: {tracker.avg_loss}")
        >>> print(f"Accuracy: {tracker.accuracy}")
    """

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        """Reset all metrics."""
        self.loss_sum = 0.0
        self.count = 0
        self.correct = 0
        self.total = 0

    def update(
        self,
        loss: float,
        correct: Optional[int] = None,
        total: Optional[int] = None,
    ) -> None:
        """
        Update metrics.

        Args:
            loss: Current loss value
            correct: Number of correct predictions (for accuracy)
            total: Total number of predictions (for accuracy)
        """
        self.loss_sum += loss
        self.count += 1
        if correct is not None:
            self.correct += correct
        if total is not None:
            self.total += total

    @property
    def avg_loss(self) -> float:
        """Get average loss."""
        return self.loss_sum / self.count if self.count > 0 else 0.0

    @property
    def accuracy(self) -> float:
        """Get accuracy."""
        return self.correct / self.total if self.total > 0 else 0.0

    def get_dict(self) -> dict[str, float]:
        """Get all metrics as dictionary."""
        return {"loss": self.avg_loss, "accuracy": self.accuracy}


def compute_accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute accuracy from model outputs and labels.

    Args:
        outputs: Model predictions (logits) of shape (batch_size, num_classes)
        labels: Ground truth labels of shape (batch_size,)

    Returns:
        Accuracy as a float between 0 and 1
    """
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    return correct / labels.size(0)


def compute_classification_metrics(
    y_true: np.ndarray | torch.Tensor,
    y_pred: np.ndarray | torch.Tensor,
    average: str = "weighted",
) -> dict[str, float]:
    """
    Compute common classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: Averaging method for multi-class ('micro', 'macro', 'weighted')

    Returns:
        Dictionary with accuracy, precision, recall, and f1
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1": f1_score(y_true, y_pred, average=average, zero_division=0),
    }


def plot_confusion_matrix(
    y_true: np.ndarray | torch.Tensor,
    y_pred: np.ndarray | torch.Tensor,
    class_names: Optional[list[str]] = None,
    save_path: Optional[str] = None,
    cmap: str = "Blues",
) -> plt.Figure:
    """
    Plot confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Optional class name labels
        save_path: Optional path to save the figure
        cmap: Colormap for the heatmap

    Returns:
        The matplotlib Figure object
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    # Show all ticks
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names if class_names else np.arange(cm.shape[1]),
        yticklabels=class_names if class_names else np.arange(cm.shape[0]),
    )

    # Rotate x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_title("Confusion Matrix")

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_training_curves(
    train_losses: list[float],
    val_losses: Optional[list[float]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot training and validation loss curves.

    Args:
        train_losses: Training losses per epoch
        val_losses: Validation losses per epoch (optional)
        save_path: Optional path to save the figure

    Returns:
        The matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, "b-", label="Training Loss", linewidth=2)

    if val_losses is not None:
        ax.plot(epochs, val_losses, "r-", label="Validation Loss", linewidth=2)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Progress")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


if __name__ == "__main__":
    # Test metrics tracker
    tracker = MetricsTracker()
    tracker.update(loss=0.5, correct=8, total=10)
    tracker.update(loss=0.3, correct=9, total=10)
    print(f"Average loss: {tracker.avg_loss:.4f}")
    print(f"Accuracy: {tracker.accuracy:.4f}")

    # Test classification metrics
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 1, 0, 2, 2])
    metrics = compute_classification_metrics(y_true, y_pred)
    print(f"\nClassification metrics: {metrics}")

    # Test plot functions
    fig = plot_training_curves([1.5, 1.0, 0.7, 0.5, 0.4], [1.6, 1.1, 0.8, 0.55, 0.45])
    fig.savefig("training_curves.png")

    fig = plot_confusion_matrix(y_true, y_pred, class_names=["Class 0", "Class 1", "Class 2"])
    fig.savefig("confusion_matrix.png")

    print("\nSaved training_curves.png and confusion_matrix.png")
