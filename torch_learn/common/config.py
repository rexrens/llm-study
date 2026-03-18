"""
Configuration management utilities.
Provides YAML-based configuration loading and validation.
"""

import os
from pathlib import Path
from typing import Any

import yaml


class Config:
    """
    Configuration manager for PyTorch experiments.

    Loads configuration from YAML files and provides attribute access.

    Examples:
        >>> config = Config("config.yaml")
        >>> print(config.learning_rate)
        >>> print(config.model.hidden_size)
    """

    def __init__(self, config_path: str | None = None, **kwargs):
        """
        Initialize configuration.

        Args:
            config_path: Path to YAML config file. If None, uses kwargs only.
            **kwargs: Additional config values that override file values.
        """
        self._config: dict[str, Any] = {}

        if config_path is not None:
            self.load(config_path)

        # Apply overrides
        self._update(kwargs)

    def load(self, config_path: str) -> None:
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to YAML config file
        """
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            if config is None:
                config = {}

        self._update(config)

    def save(self, config_path: str) -> None:
        """
        Save current configuration to YAML file.

        Args:
            config_path: Path to save config file
        """
        path = Path(config_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self._config, f, default_flow_style=False)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key (supports nested keys with dots).

        Args:
            key: Configuration key (e.g., "model.hidden_size")
            default: Default value if key not found

        Returns:
            The configuration value or default
        """
        keys = key.split(".")
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value by key (supports nested keys with dots).

        Args:
            key: Configuration key (e.g., "model.hidden_size")
            value: Value to set
        """
        keys = key.split(".")
        config = self._config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def _update(self, config: dict[str, Any]) -> None:
        """
        Update configuration with a dictionary.

        Args:
            config: Dictionary to update with
        """
        def _deep_update(d: dict, u: dict) -> dict:
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = _deep_update(d.get(k, {}), v)
                else:
                    d[k] = v
            return d

        self._config = _deep_update(self._config, config)

    def to_dict(self) -> dict[str, Any]:
        """Return configuration as dictionary."""
        return self._config.copy()

    def __getattr__(self, key: str) -> Any:
        if key.startswith("_"):
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{key}'")
        if key in self._config:
            return self._config[key]
        raise AttributeError(f"Config has no attribute '{key}'")

    def __setattr__(self, key: str, value: Any) -> None:
        if key.startswith("_"):
            super().__setattr__(key, value)
        else:
            self._config[key] = value

    def __repr__(self) -> str:
        return f"Config({self._config})"


def get_default_config() -> Config:
    """
    Get default configuration for common training scenarios.

    Returns:
        Config: Default configuration
    """
    return Config(
        model={
            "hidden_size": 128,
            "num_layers": 2,
        },
        training={
            "batch_size": 32,
            "learning_rate": 0.001,
            "num_epochs": 10,
        },
        data={
            "num_workers": 4,
            "pin_memory": True,
        },
    )


if __name__ == "__main__":
    # Test configuration
    config = Config(**get_default_config().to_dict())

    print("Default config:")
    print(config)

    # Test nested access
    print(f"\nLearning rate: {config.get('training.learning_rate')}")
    print(f"Hidden size: {config.get('model.hidden_size')}")

    # Test setting values
    config.set("training.learning_rate", 0.01)
    print(f"\nUpdated learning rate: {config.get('training.learning_rate')}")

    # Test save/load
    config.save("test_config.yaml")
    print("\nSaved config to test_config.yaml")

    loaded = Config("test_config.yaml")
    print(f"Loaded config: {loaded}")
