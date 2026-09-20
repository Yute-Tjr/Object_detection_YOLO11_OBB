from enum import StrEnum


class TaskStatus(StrEnum):
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    partial_failed = "partial_failed"
    failed = "failed"


class ImageStatus(StrEnum):
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"


class ImageStage(StrEnum):
    pending = "pending"
    object_detection = "object_detection"
    anomaly_classification = "anomaly_classification"
    rendering = "rendering"
    complete = "complete"


class OverallResult(StrEnum):
    ok = "OK"
    ng = "NG"
    unknown = "UNKNOWN"


class ImageOutcome(StrEnum):
    succeeded = "succeeded"
    failed = "failed"


TERMINAL_TASK_STATUSES = {
    TaskStatus.succeeded,
    TaskStatus.partial_failed,
    TaskStatus.failed,
}

_TASK_TRANSITIONS = {
    TaskStatus.queued: {TaskStatus.running, TaskStatus.failed},
    TaskStatus.running: {
        TaskStatus.queued,
        TaskStatus.succeeded,
        TaskStatus.partial_failed,
        TaskStatus.failed,
    },
}


def ensure_task_transition(current: TaskStatus, target: TaskStatus) -> None:
    if target not in _TASK_TRANSITIONS.get(current, set()):
        raise ValueError(f"invalid task transition: {current} -> {target}")


def aggregate_task_status(outcomes: list[ImageOutcome]) -> TaskStatus:
    successes = outcomes.count(ImageOutcome.succeeded)
    failures = outcomes.count(ImageOutcome.failed)
    if successes and failures:
        return TaskStatus.partial_failed
    return TaskStatus.succeeded if successes else TaskStatus.failed


def aggregate_image_result(labels: dict[str, str | None]) -> OverallResult:
    values = {value for value in labels.values() if value}
    if "NG" in values:
        return OverallResult.ng
    if labels.get("label3") == "OK" and labels.get("label5") == "OK":
        return OverallResult.ok
    return OverallResult.unknown
