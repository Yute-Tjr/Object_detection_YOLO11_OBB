import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from terminal_web.domain import ImageStage, OverallResult
from terminal_web.inference.color import NullColorClassifier
from terminal_web.inference.pipeline import InferenceArtifacts, LoadedInferencePipeline
from terminal_web.inference.types import ClassificationPrediction
from yolo11_obb.pipeline_predict import PipelineDetection


POINTS = ((10.0, 10.0), (50.0, 10.0), (50.0, 50.0), (10.0, 50.0))


class FakeDetector:
    def __init__(self, detections):
        self.detections = detections

    def predict(self, image_path: Path):
        return self.detections


class FakeClassifier:
    def __init__(self, label: str, fail: bool = False):
        self.label = label
        self.fail = fail
        self.calls = 0

    def predict(self, crop: np.ndarray):
        self.calls += 1
        if self.fail:
            raise RuntimeError("classifier exploded")
        return ClassificationPrediction(
            label=self.label,
            confidence=0.9,
            probabilities={"NG": 0.1, "OK": 0.9},
        )


class InferencePipelineTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.image_path = self.root / "terminal.png"
        self.assertTrue(
            cv2.imwrite(str(self.image_path), np.full((80, 80, 3), 255, np.uint8))
        )

    def tearDown(self):
        self.temp.cleanup()

    def detection(self, index: int, label: str, confidence: float):
        return PipelineDetection(
            self.image_path,
            self.image_path.name,
            index,
            label,
            confidence,
            POINTS,
        )

    def pipeline(self, detections, label3="OK", label5="OK", label3_fail=False):
        classifiers = {
            "label3": FakeClassifier(label3, fail=label3_fail),
            "label5": FakeClassifier(label5),
        }
        pipeline = LoadedInferencePipeline(
            detector=FakeDetector(detections),
            classifiers=classifiers,
            color_classifier=NullColorClassifier(),
        )
        return pipeline, classifiers

    def test_stage_order_and_unsupported_detection_are_preserved(self):
        pipeline, _ = self.pipeline(
            [
                self.detection(0, "label2", 0.8),
                self.detection(1, "label3", 0.9),
                self.detection(2, "label5", 0.85),
            ]
        )
        stages = []

        prediction = pipeline.predict_image(
            self.image_path,
            InferenceArtifacts(self.root / "crops"),
            stages.append,
        )

        self.assertEqual(
            stages,
            [
                ImageStage.object_detection,
                ImageStage.anomaly_classification,
                ImageStage.rendering,
                ImageStage.complete,
            ],
        )
        unsupported = prediction.regions[0]
        self.assertEqual(unsupported.region_label, "label2")
        self.assertIsNone(unsupported.anomaly)
        self.assertFalse(unsupported.selected_for_classification)
        self.assertEqual(prediction.overall_result, OverallResult.ok)

    def test_duplicate_label_classifies_only_highest_confidence(self):
        pipeline, classifiers = self.pipeline(
            [
                self.detection(0, "label3", 0.4),
                self.detection(1, "label3", 0.9),
                self.detection(2, "label5", 0.8),
            ]
        )

        prediction = pipeline.predict_image(
            self.image_path, InferenceArtifacts(self.root / "crops"), lambda stage: None
        )

        label3_regions = [r for r in prediction.regions if r.region_label == "label3"]
        self.assertIsNone(label3_regions[0].anomaly)
        self.assertIsNotNone(label3_regions[1].anomaly)
        self.assertEqual(classifiers["label3"].calls, 1)
        self.assertIn("duplicate label3 count=2", prediction.warnings)

    def test_missing_label5_with_label3_ok_is_unknown(self):
        pipeline, _ = self.pipeline([self.detection(0, "label3", 0.9)])
        prediction = pipeline.predict_image(
            self.image_path, InferenceArtifacts(self.root / "crops"), lambda stage: None
        )
        self.assertEqual(prediction.overall_result, OverallResult.unknown)

    def test_label3_ng_dominates_missing_label5(self):
        pipeline, _ = self.pipeline(
            [self.detection(0, "label3", 0.9)], label3="NG"
        )
        prediction = pipeline.predict_image(
            self.image_path, InferenceArtifacts(self.root / "crops"), lambda stage: None
        )
        self.assertEqual(prediction.overall_result, OverallResult.ng)

    def test_classifier_exception_becomes_region_error_and_unknown(self):
        pipeline, _ = self.pipeline(
            [
                self.detection(0, "label3", 0.9),
                self.detection(1, "label5", 0.8),
            ],
            label3_fail=True,
        )
        prediction = pipeline.predict_image(
            self.image_path, InferenceArtifacts(self.root / "crops"), lambda stage: None
        )
        label3 = next(r for r in prediction.regions if r.region_label == "label3")
        self.assertIsNone(label3.anomaly)
        self.assertIn("classifier exploded", label3.error)
        self.assertIsNone(label3.color)
        self.assertEqual(prediction.overall_result, OverallResult.unknown)
