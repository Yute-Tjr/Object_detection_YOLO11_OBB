from typing import Protocol

import numpy as np

from terminal_web.inference.types import ClassificationPrediction


class ColorClassifier(Protocol):
    def predict(
        self, crop: np.ndarray, region_label: str
    ) -> ClassificationPrediction | None:
        ...


class NullColorClassifier:
    def predict(
        self, crop: np.ndarray, region_label: str
    ) -> ClassificationPrediction | None:
        return None
