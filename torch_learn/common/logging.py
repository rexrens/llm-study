"""
Logging utilities for PyTorch experiments.
Provides structured logging with console and file output.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "torch_learn",
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    console: bool = True,
) -> logging.Logger:
    """
    Set up a logger with console and optional file handlers.

    Args:
        name: Logger name
        log_file: Path to log file. If None, no file handler is added.
        level: Logging level (e.g., logging.INFO, logging.DEBUG)
        console: Whether to add console handler

    Returns:
        Configured logger instance

    Examples:
        >>> logger = setup_logger("train", "train.log")
        >>> logger.info("Training started")
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Add console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # Add file handler
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class TrainLogger:
    """
    Training progress logger with epoch-level metrics.

    Examples:
        >>> logger = TrainLogger("training.log")
        >>> logger.log_epoch(1, loss=0.5, accuracy=0.85)
        >>> logger.log_epoch(2, loss=0.3, accuracy=0.92)
        >>> logger.close()
    """

    def __init__(self, log_file: str):
        """
        Initialize training logger.

        Args:
            log_file: Path to log file
        """
        self.logger = setup_logger("train", log_file)
        self.log_file = log_file

    def log_epoch(
        self,
        epoch: int,
        train_loss: Optional[float] = None,
        train_acc: Optional[float] = None,
        val_loss: Optional[float] = None,
        val_acc: Optional[float] = None,
        **metrics: float,
    ) -> None:
        """
        Log epoch metrics.

        Args:
            epoch: Current epoch number
            train_loss: Training loss
            train_acc: Training accuracy
            val_loss: Validation loss
            val_acc: Validation accuracy
            **metrics: Additional metrics to log
        """
        parts = [f"Epoch {epoch}"]

        if train_loss is not None:
            parts.append(f"train_loss={train_loss:.4f}")
        if train_acc is not None:
            parts.append(f"train_acc={train_acc:.4f}")
        if val_loss is not None:
            parts.append(f"val_loss={val_loss:.4f}")
        if val_acc is not None:
            parts.append(f"val_acc={val_acc:.4f}")

        for k, v in metrics.items():
            parts.append(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}")

        self.logger.info(" | ".join(parts))

    def close(self) -> None:
        """Close all handlers."""
        for handler in self.logger.handlers:
            handler.close()


class ProgressLogger:
    """
    Simple progress logger for loops.

    Examples:
        >>> logger = ProgressLogger("Processing", total=1000)
        >>> for i in range(1000):
        ...     # do work
        ...     logger.update(i)
        >>> logger.finish()
    """

    def __init__(self, name: str, total: int):
        """
        Initialize progress logger.

        Args:
            name: Name of the operation
            total: Total number of items
        """
        self.name = name
        self.total = total
        self.count = 0
        self.logger = logging.getLogger("progress")

    def update(self, batch_idx: int) -> None:
        """
        Update progress.

        Args:
            batch_idx: Current batch index
        """
        self.count = batch_idx + 1
        if self.count % max(1, self.total // 10) == 0 or self.count == self.total:
            progress = 100 * self.count / self.total
            self.logger.info(f"{self.name}: {progress:.0f}% ({self.count}/{self.total})")

    def finish(self) -> None:
        """Mark progress as complete."""
        self.logger.info(f"{self.name}: Complete ({self.total}/{self.total})")


if __name__ == "__main__":
    # Test logger
    logger = setup_logger("test", "test.log", level=logging.DEBUG)

    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")

    # Test training logger
    train_logger = TrainLogger("train.log")
    train_logger.log_epoch(1, train_loss=1.5, train_acc=0.5)
    train_logger.log_epoch(2, train_loss=1.0, train_acc=0.7, val_loss=0.9, val_acc=0.68)
    train_logger.log_epoch(3, train_loss=0.7, train_acc=0.82, val_loss=0.75, val_acc=0.8)
    train_logger.close()

    # Test progress logger
    progress = ProgressLogger("Training", 100)
    for i in range(0, 100, 20):
        progress.update(i)
    progress.finish()
