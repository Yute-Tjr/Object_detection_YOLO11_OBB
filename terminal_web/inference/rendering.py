from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from terminal_web.inference.types import ImagePrediction, RegionPrediction


OK_COLOR = (0, 180, 0)
NG_COLOR = (0, 0, 255)
UNSUPPORTED_COLOR = (128, 128, 128)

_CJK_FONT_CANDIDATES = (
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf"),
    Path("/System/Library/Fonts/PingFang.ttc"),
    Path("/System/Library/Fonts/STHeiti Light.ttc"),
)


def format_region_label(
    region: RegionPrediction,
    *,
    ascii_fallback: bool = False,
) -> str:
    separator = " - " if ascii_fallback else " · "
    if not region.selected_for_classification:
        status = "classification unavailable" if ascii_fallback else "暂不支持分类"
        return separator.join((region.region_label, status))
    if region.error or region.anomaly is None:
        status = "classification failed" if ascii_fallback else "分类失败"
        return separator.join((region.region_label, status))

    parts = [region.region_label, region.anomaly.label]
    if region.color is not None:
        parts.append(region.color.label)
    return separator.join(parts)


def region_color(region: RegionPrediction) -> tuple[int, int, int]:
    if region.anomaly is not None and region.anomaly.label == "NG":
        return NG_COLOR
    if region.anomaly is not None and region.anomaly.label == "OK":
        return OK_COLOR
    return UNSUPPORTED_COLOR


def label_origin(
    points,
    label_width: int,
    label_height: int,
    image_width: int,
    image_height: int,
    gap: int = 8,
) -> tuple[int, int]:
    left = max(0, int(min(x for x, _ in points)))
    right = min(image_width, int(max(x for x, _ in points)))
    top = max(0, int(min(y for _, y in points)))
    bottom = min(image_height, int(max(y for _, y in points)))

    x = right + gap
    if x + label_width > image_width:
        x = left - gap - label_width
    x = min(max(x, 0), max(image_width - label_width, 0))

    y = int((top + bottom - label_height) / 2)
    y = min(max(y, 0), max(image_height - label_height, 0))
    return x, y


def _load_font(configured: Path | None, size: int):
    candidates = ((configured,) if configured else ()) + _CJK_FONT_CANDIDATES
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            try:
                return ImageFont.truetype(str(candidate), size=size), True
            except OSError:
                continue
    return ImageFont.load_default(), False


def render_prediction(
    source: Path,
    prediction: ImagePrediction,
    output: Path,
    *,
    cjk_font_path: Path | None = None,
    line_thickness: int = 2,
) -> None:
    image = cv2.imread(str(source))
    if image is None:
        raise ValueError(f"failed to read image: {source}")

    for region in prediction.regions:
        points = np.asarray(region.points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(
            image,
            [points],
            isClosed=True,
            color=region_color(region),
            thickness=line_thickness,
            lineType=cv2.LINE_AA,
        )

    font, supports_cjk = _load_font(cjk_font_path, size=18)
    canvas = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(canvas)
    for region in prediction.regions:
        color_bgr = region_color(region)
        color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
        text = format_region_label(region, ascii_fallback=not supports_cjk)
        text_box = draw.textbbox((0, 0), text, font=font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        label_width = text_width + 12
        label_height = text_height + 8
        x, y = label_origin(
            region.points,
            label_width,
            label_height,
            canvas.width,
            canvas.height,
        )
        draw.rounded_rectangle(
            (x, y, x + label_width, y + label_height),
            radius=3,
            fill=color_rgb,
        )
        draw.text((x + 6, y + 3), text, font=font, fill=(255, 255, 255))

    rendered = cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), rendered):
        raise RuntimeError(f"failed to write visualization: {output}")
