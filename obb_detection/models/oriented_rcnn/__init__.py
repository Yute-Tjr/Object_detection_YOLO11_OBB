"""Oriented R-CNN R50-FPN integration for MMRotate 1.x."""

from .config import (
    DEFAULT_BASE_CONFIG,
    find_base_config,
    load_class_names,
    render_oriented_rcnn_config,
    validate_dota_dataset,
)

__all__ = [
    "DEFAULT_BASE_CONFIG",
    "find_base_config",
    "load_class_names",
    "render_oriented_rcnn_config",
    "validate_dota_dataset",
]

