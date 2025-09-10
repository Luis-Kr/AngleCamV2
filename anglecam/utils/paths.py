from pathlib import Path
from typing import Union


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent.absolute()


def get_anglecam_root() -> Path:
    """Get the anglecam package root directory."""
    return Path(__file__).parent.parent.absolute()


def resolve_data_path(relative_path: Union[str, Path]) -> Path:
    """Resolve a data path relative to project root."""
    return get_project_root() / relative_path


def resolve_config_path(relative_path: Union[str, Path]) -> Path:
    """Resolve a config path relative to anglecam package."""
    return get_anglecam_root() / relative_path
