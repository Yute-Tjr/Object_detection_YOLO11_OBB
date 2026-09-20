from terminal_web.models import Detection, InspectionImage, InspectionTask
from terminal_web.schemas import (
    ClassificationResponse,
    DetectionResponse,
    ImageDetail,
    ImageSummary,
    TaskDetail,
    TaskSummary,
)


def image_summary(image: InspectionImage) -> ImageSummary:
    return ImageSummary(
        id=image.id,
        sequence_no=image.sequence_no,
        original_filename=image.original_filename,
        status=image.status,
        stage=image.stage,
        overall_result=image.overall_result,
        width=image.width,
        height=image.height,
        size_bytes=image.size_bytes,
        original_url=f"/api/v1/images/{image.id}/original",
        result_url=(f"/api/v1/images/{image.id}/result" if image.result_path else None),
        error_code=image.error_code,
        error_message=image.error_message,
    )


def task_summary(task: InspectionTask) -> TaskSummary:
    detector_name = task.detector_model.name if task.detector_model else "YOLO11l-OBB"
    return TaskSummary(
        id=task.id,
        display_id=task.display_id,
        name=task.name,
        operator=task.operator,
        note=task.note,
        status=task.status,
        current_stage=task.current_stage,
        total_images=task.total_images,
        completed_images=task.completed_images,
        succeeded_images=task.succeeded_images,
        failed_images=task.failed_images,
        detector_model=detector_name,
        created_at=task.created_at,
        started_at=task.started_at,
        finished_at=task.finished_at,
    )


def task_detail(task: InspectionTask) -> TaskDetail:
    summary = task_summary(task)
    return TaskDetail(
        **summary.model_dump(),
        images=[image_summary(image) for image in task.images],
    )


def detection_response(detection: Detection) -> DetectionResponse:
    classifications = {
        result.classifier_type: ClassificationResponse(
            classifier_type=result.classifier_type,
            predicted_label=result.predicted_label,
            confidence=result.confidence,
            probabilities=result.probabilities,
            model_name=result.model.name if result.model else None,
            model_version=result.model.version if result.model else None,
        )
        for result in detection.classifications
    }
    return DetectionResponse(
        id=detection.id,
        region_label=detection.region_label,
        confidence=detection.confidence,
        points=detection.points,
        selected_for_classification=detection.selected_for_classification,
        crop_url=None,
        anomaly=classifications.get("anomaly"),
        color=classifications.get("color"),
    )


def image_detail(image: InspectionImage) -> ImageDetail:
    summary = image_summary(image)
    return ImageDetail(
        **summary.model_dump(),
        detections=[detection_response(item) for item in image.detections],
    )
