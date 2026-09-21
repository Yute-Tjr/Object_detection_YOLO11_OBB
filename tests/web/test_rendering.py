import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from terminal_web.domain import OverallResult
from terminal_web.inference import rendering
from terminal_web.inference.rendering import (
    NG_COLOR,
    OK_COLOR,
    UNSUPPORTED_COLOR,
    annotation_style,
    format_region_label,
    label_origin,
    render_prediction,
)
from terminal_web.inference.types import (
    ClassificationPrediction,
    ImagePrediction,
    RegionPrediction,
)


def classification(label: str) -> ClassificationPrediction:
    return ClassificationPrediction(label, 0.95, {label: 0.95})


def region(
    label: str,
    points,
    *,
    anomaly=None,
    color=None,
    selected=True,
    error=None,
) -> RegionPrediction:
    return RegionPrediction(
        region_label=label,
        detection_confidence=0.9,
        points=points,
        selected_for_classification=selected,
        anomaly=anomaly,
        color=color,
        crop_path=None,
        error=error,
    )


class RenderingTest(unittest.TestCase):
    def test_annotation_style_scales_for_tall_factory_images(self):
        self.assertEqual(annotation_style(506, 1627), (30, 3))
        self.assertEqual(annotation_style(1440, 3072), (54, 6))

    def test_fallback_font_keeps_requested_readable_size(self):
        with patch.object(rendering, "_CJK_FONT_CANDIDATES", ()):
            font, supports_cjk = rendering._load_font(None, 30)

        text_box = font.getbbox("label3 - NG")
        self.assertFalse(supports_cjk)
        self.assertGreaterEqual(text_box[3] - text_box[1], 20)

    def test_label_origin_prefers_right_side(self):
        self.assertEqual(
            label_origin(
                ((10, 20), (60, 20), (60, 80), (10, 80)),
                40,
                18,
                200,
                120,
            ),
            (68, 41),
        )

    def test_label_origin_falls_back_left_and_clamps_to_canvas(self):
        x, y = label_origin(
            ((150, 0), (195, 0), (195, 40), (150, 40)),
            100,
            24,
            200,
            80,
        )

        self.assertGreaterEqual(x, 0)
        self.assertLessEqual(x + 100, 200)
        self.assertGreaterEqual(y, 0)
        self.assertLessEqual(y + 24, 80)
        self.assertLess(x, 150)

    def test_format_labels_match_factory_semantics(self):
        square = ((0, 0), (10, 0), (10, 10), (0, 10))
        self.assertEqual(
            format_region_label(region("label3", square, anomaly=classification("OK"))),
            "label3 · OK",
        )
        self.assertEqual(
            format_region_label(
                region(
                    "label5",
                    square,
                    anomaly=classification("NG"),
                    color=classification("蓝色"),
                )
            ),
            "label5 · NG · 蓝色",
        )
        self.assertEqual(
            format_region_label(region("label2", square, selected=False)),
            "label2 · 暂不支持分类",
        )
        self.assertEqual(
            format_region_label(region("label3", square, error="failed")),
            "label3 · 分类失败",
        )

    def test_border_pixels_use_green_red_and_gray(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.png"
            output = root / "result.png"
            self.assertTrue(
                cv2.imwrite(str(source), np.full((160, 180, 3), 255, np.uint8))
            )
            regions = (
                region(
                    "label3",
                    ((10, 10), (60, 10), (60, 50), (10, 50)),
                    anomaly=classification("OK"),
                ),
                region(
                    "label5",
                    ((90, 10), (150, 10), (150, 50), (90, 50)),
                    anomaly=classification("NG"),
                ),
                region(
                    "label2",
                    ((10, 90), (60, 90), (60, 140), (10, 140)),
                    selected=False,
                ),
            )
            prediction = ImagePrediction(regions, OverallResult.ng, ())

            render_prediction(source, prediction, output)

            rendered = cv2.imread(str(output))
            self.assertEqual(tuple(rendered[50, 35]), OK_COLOR)
            self.assertEqual(tuple(rendered[50, 120]), NG_COLOR)
            self.assertEqual(tuple(rendered[140, 35]), UNSUPPORTED_COLOR)
