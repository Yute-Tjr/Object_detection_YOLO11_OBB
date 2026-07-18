"""Compatibility imports for historical RHINO commands.

The conversion is MMRotate-generic and now lives in
``obb_detection.common.mmrotate_predictions``.
"""

from obb_detection.common.mmrotate_predictions import (
    index_images_by_stem,
    rbox_to_corners,
    rboxes_to_yolo_lines,
)

__all__ = ["index_images_by_stem", "rbox_to_corners", "rboxes_to_yolo_lines"]
