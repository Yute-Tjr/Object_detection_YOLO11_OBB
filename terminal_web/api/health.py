from pathlib import Path

from fastapi import APIRouter, Depends

from terminal_web.api.dependencies import ReadinessProvider, get_readiness
from terminal_web.schemas import HealthResponse


router = APIRouter(tags=["health"])


class FileReadinessProvider:
    def __init__(
        self,
        detector_weights: Path,
        label3_weights: Path,
        label5_weights: Path,
    ):
        self._models = [
            ("detector", "YOLO11l-OBB", "baseline", detector_weights),
            ("anomaly", "ResNet18-label3", "best", label3_weights),
            ("anomaly", "ResNet18-label5", "best", label5_weights),
        ]

    def snapshot(self) -> HealthResponse:
        models_ready = all(path.is_file() for _, _, _, path in self._models)
        return HealthResponse(
            api_ready=True,
            database_ready=True,
            worker_ready=True,
            models_ready=models_ready,
        )


@router.get("/health", response_model=HealthResponse)
def health(
    readiness: ReadinessProvider = Depends(get_readiness),
) -> HealthResponse:
    return readiness.snapshot()
