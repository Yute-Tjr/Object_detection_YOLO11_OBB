from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
import torch
from PIL import Image

from terminal_web.domain import ImageStage, aggregate_image_result
from terminal_web.inference.color import ColorClassifier, NullColorClassifier
from terminal_web.inference.types import (
    ClassificationPrediction,
    ImagePrediction,
    RegionPrediction,
)
from yolo11_obb.obb_crop import rectify_obb_crop
from yolo11_obb.pipeline_predict import (
    PipelineClassifier,
    PipelineDetection,
    _load_yolo_model,
    detections_from_yolo_result,
    load_pipeline_classifiers,
    selected_detection_by_label,
)


class DetectorAdapter(Protocol):
    def predict(self, image_path: Path) -> Sequence[PipelineDetection]:
        ...


class AnomalyClassifier(Protocol):
    def predict(self, crop: np.ndarray) -> ClassificationPrediction:
        ...


@dataclass(frozen=True)
class InferenceArtifacts:
    crop_dir: Path
    result_path: Path | None = None

    def crop_path(self, region_label: str, detection_index: int) -> Path:
        return self.crop_dir / f"{region_label}_{detection_index}.png"


class YoloDetectorAdapter:
    def __init__(
        self,
        model,
        *,
        imgsz: int,
        confidence: float,
        device: str | None,
    ):
        self.model = model
        self.imgsz = imgsz
        self.confidence = confidence
        self.device = device

    def predict(self, image_path: Path) -> Sequence[PipelineDetection]:
        kwargs = {
            "source": str(image_path),
            "imgsz": self.imgsz,
            "conf": self.confidence,
            "save": False,
            "verbose": False,
        }
        if self.device is not None:
            kwargs["device"] = self.device
        results = self.model.predict(**kwargs)
        if not results:
            return []
        return detections_from_yolo_result(results[0])


class TorchClassifierAdapter:
    def __init__(self, classifier: PipelineClassifier):
        self.classifier = classifier

    def predict(self, crop: np.ndarray) -> ClassificationPrediction:
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        tensor = self.classifier.transform(pil_image).unsqueeze(0).to(  # type: ignore[operator]
            self.classifier.device
        )
        with torch.no_grad():
            output = self.classifier.model(tensor)
            probabilities = torch.softmax(output, dim=1).detach().cpu().tolist()[0]
        predicted_index = max(range(len(probabilities)), key=probabilities.__getitem__)
        return ClassificationPrediction(
            label=self.classifier.classes[predicted_index],
            confidence=float(probabilities[predicted_index]),
            probabilities={
                name: float(probability)
                for name, probability in zip(self.classifier.classes, probabilities)
            },
        )


class LoadedInferencePipeline:
    required_labels = ("label3", "label5")

    def __init__(
        self,
        *,
        detector_weights: Path | None = None,
        label3_weights: Path | None = None,
        label5_weights: Path | None = None,
        detection_imgsz: int = 1280,
        detection_confidence: float = 0.25,
        detection_device: str | None = None,
        classification_imgsz: int = 224,
        classification_device: str | None = None,
        detector: DetectorAdapter | None = None,
        classifiers: Mapping[str, AnomalyClassifier] | None = None,
        color_classifier: ColorClassifier | None = None,
    ):
        if detector is None:
            if detector_weights is None:
                raise ValueError("detector_weights is required")
            detector = YoloDetectorAdapter(
                _load_yolo_model(Path(detector_weights).expanduser().resolve()),
                imgsz=detection_imgsz,
                confidence=detection_confidence,
                device=detection_device,
            )
        if classifiers is None:
            if label3_weights is None or label5_weights is None:
                raise ValueError("label3_weights and label5_weights are required")
            loaded = load_pipeline_classifiers(
                {
                    "label3": Path(label3_weights).expanduser().resolve(),
                    "label5": Path(label5_weights).expanduser().resolve(),
                },
                device_name=classification_device,
                imgsz=classification_imgsz,
            )
            classifiers = {
                label: TorchClassifierAdapter(classifier)
                for label, classifier in loaded.items()
            }
        missing = set(self.required_labels) - set(classifiers)
        if missing:
            raise ValueError(f"missing classifiers: {', '.join(sorted(missing))}")

        self.detector = detector
        self.classifiers = dict(classifiers)
        self.color_classifier = color_classifier or NullColorClassifier()

    def predict_image(
        self,
        image_path: Path,
        artifacts: InferenceArtifacts,
        stage_callback: Callable[[ImageStage], None],
    ) -> ImagePrediction:
        image_path = Path(image_path).expanduser().resolve()
        stage_callback(ImageStage.object_detection)
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"failed to read image: {image_path}")
        detections = list(self.detector.predict(image_path))
        selected, selection_warnings = selected_detection_by_label(
            detections, self.required_labels
        )

        stage_callback(ImageStage.anomaly_classification)
        warnings = list(selection_warnings)
        regions: list[RegionPrediction] = []
        labels: dict[str, str | None] = {label: None for label in self.required_labels}
        for detection in detections:
            is_selected = selected.get(detection.det_label) is detection
            anomaly = None
            color = None
            crop_path = None
            error = None
            if is_selected:
                try:
                    crop = rectify_obb_crop(image, detection.points)
                    destination = artifacts.crop_path(
                        detection.det_label, detection.det_index
                    )
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    if not cv2.imwrite(str(destination), crop):
                        raise RuntimeError(f"failed to write crop: {destination.name}")
                    crop_path = str(destination)
                    anomaly = self.classifiers[detection.det_label].predict(crop)
                    labels[detection.det_label] = anomaly.label
                    color = self.color_classifier.predict(crop, detection.det_label)
                except Exception as exc:
                    error = str(exc)
                    warnings.append(
                        f"{detection.det_label} classification failed: {error}"
                    )
            regions.append(
                RegionPrediction(
                    region_label=detection.det_label,
                    detection_confidence=detection.det_conf,
                    points=tuple(detection.points),
                    selected_for_classification=is_selected,
                    anomaly=anomaly,
                    color=color,
                    crop_path=crop_path,
                    error=error,
                )
            )

        stage_callback(ImageStage.rendering)
        overall_result = aggregate_image_result(labels)
        prediction = ImagePrediction(
            regions=tuple(regions),
            overall_result=overall_result,
            warnings=tuple(warnings),
        )
        stage_callback(ImageStage.complete)
        return prediction
