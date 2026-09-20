import unittest

from terminal_web.domain import (
    ImageOutcome,
    OverallResult,
    TaskStatus,
    aggregate_image_result,
    aggregate_task_status,
    ensure_task_transition,
)


class DomainTest(unittest.TestCase):
    def test_partial_failure_is_not_success(self):
        status = aggregate_task_status([ImageOutcome.succeeded, ImageOutcome.failed])
        self.assertEqual(status, TaskStatus.partial_failed)

    def test_missing_classifier_result_is_unknown(self):
        self.assertEqual(
            aggregate_image_result({"label3": "OK", "label5": None}),
            OverallResult.unknown,
        )

    def test_explicit_ng_dominates_missing_region(self):
        self.assertEqual(
            aggregate_image_result({"label3": "NG", "label5": None}),
            OverallResult.ng,
        )

    def test_terminal_task_cannot_return_to_running(self):
        with self.assertRaisesRegex(ValueError, "invalid task transition"):
            ensure_task_transition(TaskStatus.succeeded, TaskStatus.running)
