from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from terminal_web.inference.types import ImagePrediction, RegionPrediction


OK_COLOR = (0, 180, 0)
NG_COLOR = (0, 0, 255)
UNSUPPORTED_COLOR = (255, 119, 22)  # RGB #1677FF expressed as OpenCV BGR.

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


def _load_font(configured: Path | None, size: int):
    candidates = ((configured,) if configured else ()) + _CJK_FONT_CANDIDATES
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            try:
                return ImageFont.truetype(str(candidate), size=size), True
            except OSError:
                continue
    try:
        return ImageFont.load_default(size=size), False
    except TypeError:  # Pillow < 10.1 does not expose the size argument.
        return ImageFont.load_default(), False


def annotation_style(image_width: int, image_height: int) -> tuple[int, int]:
    """Return a readable font size and outline width for the source image."""
    short_side = min(image_width, image_height)
    font_size = min(54, max(30, round(short_side * 0.06)))
    line_thickness = min(6, max(2, round(short_side * 0.006)))
    return font_size, line_thickness


def _layout_label_tops(
    target_centers: list[int],
    label_heights: list[int],
    canvas_height: int,
    *,
    gap: int,
    margin: int,
) -> list[int]:
    order = sorted(range(len(target_centers)), key=target_centers.__getitem__)
    tops = [0] * len(order)
    cursor = margin

    for index in order:
        desired = target_centers[index] - label_heights[index] // 2
        tops[index] = max(desired, cursor)
        cursor = tops[index] + label_heights[index] + gap

    if order:
        overflow = tops[order[-1]] + label_heights[order[-1]] + margin - canvas_height
        if overflow > 0:
            tops[order[-1]] -= overflow
            for position in range(len(order) - 2, -1, -1):
                index = order[position]
                next_index = order[position + 1]
                tops[index] = min(
                    tops[index],
                    tops[next_index] - gap - label_heights[index],
                )

    return tops


def render_prediction(
    source: Path,
    prediction: ImagePrediction,
    output: Path,
    *,
    cjk_font_path: Path | None = None,
    line_thickness: int | None = None,
) -> None:
    image = cv2.imread(str(source))
    if image is None:
        raise ValueError(f"failed to read image: {source}")

    font_size, automatic_thickness = annotation_style(image.shape[1], image.shape[0])
    line_thickness = line_thickness or automatic_thickness

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

    font, supports_cjk = _load_font(cjk_font_path, size=font_size)
    image_rgb = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    measurement_draw = ImageDraw.Draw(image_rgb)
    horizontal_padding = max(8, round(font_size * 0.34))
    vertical_padding = max(6, round(font_size * 0.22))
    gutter_gap = max(12, round(font_size * 0.5))
    label_gap = max(6, round(font_size * 0.25))
    label_specs = []

    for region in prediction.regions:
        text = format_region_label(region, ascii_fallback=not supports_cjk)
        text_box = measurement_draw.textbbox((0, 0), text, font=font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        label_width = text_width + horizontal_padding * 2
        label_height = text_height + vertical_padding * 2
        target_center = round(
            (min(y for _, y in region.points) + max(y for _, y in region.points)) / 2
        )
        label_specs.append(
            (region, text, text_box, label_width, label_height, target_center)
        )

    maximum_label_width = max((spec[3] for spec in label_specs), default=0)
    minimum_label_height = (
        sum(spec[4] for spec in label_specs)
        + label_gap * max(len(label_specs) - 1, 0)
        + gutter_gap * 2
    )
    canvas_height = max(image_rgb.height, minimum_label_height)
    canvas_width = image_rgb.width + maximum_label_width + gutter_gap * 2
    canvas = Image.new("RGB", (canvas_width, canvas_height), (247, 249, 252))
    canvas.paste(image_rgb, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.line(
        (image_rgb.width, 0, image_rgb.width, canvas_height),
        fill=(218, 224, 233),
        width=max(1, line_thickness),
    )

    label_tops = _layout_label_tops(
        [spec[5] for spec in label_specs],
        [spec[4] for spec in label_specs],
        canvas_height,
        gap=label_gap,
        margin=gutter_gap,
    )
    label_x = image_rgb.width + gutter_gap

    for spec, y in zip(label_specs, label_tops):
        region, text, text_box, label_width, label_height, _ = spec
        color_bgr = region_color(region)
        color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
        label_center = y + label_height // 2
        draw.line(
            (image_rgb.width, label_center, label_x, label_center),
            fill=color_rgb,
            width=max(1, line_thickness),
        )
        draw.rounded_rectangle(
            (label_x, y, label_x + label_width, y + label_height),
            radius=max(3, round(font_size * 0.16)),
            fill=color_rgb,
        )
        draw.text(
            (
                label_x + horizontal_padding,
                y + vertical_padding - text_box[1],
            ),
            text,
            font=font,
            fill=(255, 255, 255),
        )

    rendered = cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), rendered):
        raise RuntimeError(f"failed to write visualization: {output}")
