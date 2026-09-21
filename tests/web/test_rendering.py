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
    annotation_style,
    format_region_label,
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

    def test_long_label_is_rendered_outside_the_source_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.png"
            output = root / "result.png"
            source_image = np.full((160, 180, 3), 255, np.uint8)
            self.assertTrue(cv2.imwrite(str(source), source_image))
            prediction = ImagePrediction(
                (
                    region(
                        "label1_thick",
                        ((50, 50), (130, 50), (130, 110), (50, 110)),
                        selected=False,
                    ),
                ),
                OverallResult.unknown,
                (),
            )

            render_prediction(source, prediction, output)

            rendered = cv2.imread(str(output))
            self.assertGreater(rendered.shape[1], source_image.shape[1])
            self.assertEqual(tuple(rendered[80, 90]), (255, 255, 255))
            gutter = rendered[:, source_image.shape[1] :]
            self.assertTrue(np.any(np.all(gutter == (255, 119, 22), axis=2)))

    def test_border_pixels_use_green_red_and_blue(self):
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
            self.assertEqual(tuple(rendered[140, 35]), (255, 119, 22))
