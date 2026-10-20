"""Configuration loading and path resolution.

Configurations are YAML files. Any value may reference an environment
variable with ``${VAR}`` or ``${VAR:default}``, which keeps machine-specific
locations such as the dataset root out of the repository::

    dataset_root: ${FEDBCA_ROOT}
    preprocessed_root: ${FEDBCA_PREPROCESSED:./data/preprocessed}

Command-line overrides take precedence over the file, and the file takes
precedence over the environment default.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import yaml

__all__ = ["load_config", "resolve_path", "expand_env", "merge_overrides"]

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::([^}]*))?\}")


def expand_env(value):
    """Recursively expand ``${VAR}`` / ``${VAR:default}`` references.

    Raises:
        KeyError: if a referenced variable is unset and has no default.
    """
    if isinstance(value, dict):
        return {k: expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_env(v) for v in value]
    if not isinstance(value, str):
        return value

    def _replace(match: re.Match) -> str:
        name, default = match.group(1), match.group(2)
        if name in os.environ:
            return os.environ[name]
        if default is not None:
            return default
        raise KeyError(
            f"Environment variable {name!r} is referenced by the configuration "
            f"but is not set, and no default was given."
        )

    return _ENV_PATTERN.sub(_replace, value)


def load_config(path: str | Path) -> dict:
    """Load a YAML configuration and expand its environment references."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}

    if not isinstance(config, dict):
        raise ValueError(f"{path} must contain a YAML mapping at the top level.")

    return expand_env(config)


def merge_overrides(config: dict, **overrides) -> dict:
    """Return ``config`` updated with every override that is not ``None``."""
    merged = dict(config)
    merged.update({k: v for k, v in overrides.items() if v is not None})
    return merged


def resolve_path(value: str | Path, key: str = "path") -> Path:
    """Expand ``~`` and environment references, and return an absolute path.

    Raises:
        ValueError: if ``value`` is empty or still contains an unexpanded
            placeholder.
    """
    if value is None or str(value).strip() == "":
        raise ValueError(f"Missing required path for {key!r}.")

    text = expand_env(str(value))
    if "${" in text:
        raise ValueError(f"Unresolved placeholder in {key!r}: {text}")

    return Path(text).expanduser().resolve()
