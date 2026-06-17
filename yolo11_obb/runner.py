from __future__ import annotations

from typing import Any

from .config import (
    EvalOptions,
    PredictOptions,
    TrainOptions,
    build_eval_kwargs,
    build_predict_kwargs,
    build_train_kwargs,
)

"""
调用 Ultralytics。
"""

def _load_yolo() -> Any:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics is not installed. Install it with: "
            "python3 -m pip install ultralytics"
        ) from exc
    return YOLO


def train(options: TrainOptions) -> Any:
    """
    训练模型入口
    :param options:
    :return:
    """
    YOLO = _load_yolo()
    model = YOLO(options.model)
    return model.train(**build_train_kwargs(options))


def evaluate(options: EvalOptions) -> Any:
    """
    评估模型入口
    :param options:
    :return:
    """
    YOLO = _load_yolo()
    model = YOLO(str(options.model))
    return model.val(**build_eval_kwargs(options))


def predict(options: PredictOptions) -> Any:
    YOLO = _load_yolo()
    model = YOLO(str(options.model))
    return model.predict(**build_predict_kwargs(options))
