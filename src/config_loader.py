"""Configuration loading utilities for training and experiments."""

from pathlib import Path
from typing import Any, Dict

import yaml


class Config:
    """Configuration container with dot-notation access.

    Allows accessing nested config values with dots,
    e.g., config.model.matern_nu

    :param config_dict: Dictionary of configuration values
    """

    def __init__(self, config_dict: Dict[str, Any]):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    def __repr__(self):
        items = []
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                items.append(f"{key}=Config(...)")
            else:
                items.append(f"{key}={value}")
        return f"Config({', '.join(items)})"

    def to_dict(self) -> Dict[str, Any]:
        """Convert Config object back to dictionary.

        :return: Dictionary representation of config
        """
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result


def load_config(config_path: str = "config/training_config.yaml") -> Config:
    """Load training configuration from YAML file.

    :param config_path: Path to YAML config file
    :return: Configuration object with dot-notation access
    :raises FileNotFoundError: If config file doesn't exist
    """
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n" f"Please create it or check the path."
        )

    with open(config_file, "r") as f:
        config_dict = yaml.safe_load(f)

    return Config(config_dict)


def load_config_with_overrides(
    config_path: str = "config/training_config.yaml", **overrides
) -> Config:
    """Load config with command-line overrides.

    Useful for quick experiments without modifying config file.

    Example:
        config = load_config_with_overrides(
            'config/training_config.yaml',
            epochs=5000,
            learning_rate=0.01
        )

    :param config_path: Path to YAML config file
    :param overrides: Key-value pairs to override config values
    :return: Configuration object with overrides applied
    """
    config = load_config(config_path)

    # Apply overrides (supports nested keys with dot notation)
    for key, value in overrides.items():
        if "." in key:
            # Handle nested keys like 'model.matern_nu'
            parts = key.split(".")
            obj = config
            for part in parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, parts[-1], value)
        else:
            setattr(config, key, value)

    return config


# Backward compatibility: Convert config to argparse-like namespace
def config_to_args(config: Config):
    """Flatten config for backward compatibility with argparse code.

    Converts nested config structure to flat namespace matching old argparse.

    :param config: Config object to flatten
    :return: Flat namespace object
    """
    flat_config = {
        # Model parameters
        "kernel": config.model.kernel,
        "matern_nu": config.model.matern_nu,
        "dim": config.model.input_dim,
        "noise": config.model.noise_prior,
        "lengthscale_prior": config.model.lengthscale_prior,
        "train_flag_ls": config.model.train_lengthscale,
        "min_ls": config.model.min_lengthscale,
        # Training parameters
        "epochs": config.training.epochs,
        "lr": config.training.learning_rate,
        "log_interval": config.training.log_interval,
        "train_flag": config.training.train_flag,
        # Grid
        "grid_dim": config.grid.grid_dim,
        # Acquisition
        "mc_an_flag": config.acquisition.mc_or_analytic,
        "acq_f": config.acquisition.functions,
        "num_suggestions": config.acquisition.num_suggestions,
        # WandB
        "wandb": config.wandb.enabled,
        # Compute
        "seed": config.compute.seed,
        "gpus": config.compute.gpus,
    }

    return Config(flat_config)
