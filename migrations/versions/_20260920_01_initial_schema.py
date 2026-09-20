"""Create the initial terminal inspection schema.

Revision ID: 20260920_01
Revises:
Create Date: 2026-09-20
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "20260920_01"
down_revision: str | None = None
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "model_registry",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("model_type", sa.String(length=32), nullable=False),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("version", sa.String(length=64), nullable=False),
        sa.Column("weights_path", sa.Text(), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column("active", sa.Boolean(), nullable=False),
        sa.Column("load_config", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("model_type", "name", "version", name="uq_model_identity"),
    )
    op.create_index("ix_model_registry_model_type", "model_registry", ["model_type"])

    op.create_table(
        "inspection_tasks",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("display_id", sa.String(length=32), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("current_stage", sa.String(length=32), nullable=False),
        sa.Column("total_images", sa.Integer(), nullable=False),
        sa.Column("completed_images", sa.Integer(), nullable=False),
        sa.Column("succeeded_images", sa.Integer(), nullable=False),
        sa.Column("failed_images", sa.Integer(), nullable=False),
        sa.Column("attempt_count", sa.Integer(), nullable=False),
        sa.Column("worker_id", sa.String(length=128), nullable=True),
        sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("detector_model_id", sa.Uuid(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("note", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["detector_model_id"], ["model_registry.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("display_id"),
    )
    op.create_index(
        "ix_inspection_tasks_status_created",
        "inspection_tasks",
        ["status", "created_at"],
    )

    op.create_table(
        "inspection_images",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("task_id", sa.Uuid(), nullable=False),
        sa.Column("sequence_no", sa.Integer(), nullable=False),
        sa.Column("original_filename", sa.String(length=255), nullable=False),
        sa.Column("stored_filename", sa.String(length=255), nullable=False),
        sa.Column("original_path", sa.Text(), nullable=False),
        sa.Column("result_path", sa.Text(), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("stage", sa.String(length=32), nullable=False),
        sa.Column("overall_result", sa.String(length=16), nullable=False),
        sa.Column("width", sa.Integer(), nullable=False),
        sa.Column("height", sa.Integer(), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("error_code", sa.String(length=64), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["task_id"], ["inspection_tasks.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("task_id", "sequence_no", name="uq_task_image_sequence"),
    )
    op.create_index("ix_inspection_images_task_id", "inspection_images", ["task_id"])
    op.create_index(
        "ix_inspection_images_original_filename",
        "inspection_images",
        ["original_filename"],
    )

    op.create_table(
        "detections",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("image_id", sa.Uuid(), nullable=False),
        sa.Column("region_label", sa.String(length=64), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("points", sa.JSON(), nullable=False),
        sa.Column("selected_for_classification", sa.Boolean(), nullable=False),
        sa.Column("crop_path", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["image_id"], ["inspection_images.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_detections_image_id", "detections", ["image_id"])

    op.create_table(
        "classification_results",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("detection_id", sa.Uuid(), nullable=False),
        sa.Column("classifier_type", sa.String(length=32), nullable=False),
        sa.Column("predicted_label", sa.String(length=64), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("probabilities", sa.JSON(), nullable=True),
        sa.Column("model_id", sa.Uuid(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["detection_id"], ["detections.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["model_id"], ["model_registry.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_classification_results_detection_id",
        "classification_results",
        ["detection_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_classification_results_detection_id",
        table_name="classification_results",
    )
    op.drop_table("classification_results")
    op.drop_index("ix_detections_image_id", table_name="detections")
    op.drop_table("detections")
    op.drop_index("ix_inspection_images_original_filename", table_name="inspection_images")
    op.drop_index("ix_inspection_images_task_id", table_name="inspection_images")
    op.drop_table("inspection_images")
    op.drop_index("ix_inspection_tasks_status_created", table_name="inspection_tasks")
    op.drop_table("inspection_tasks")
    op.drop_index("ix_model_registry_model_type", table_name="model_registry")
    op.drop_table("model_registry")
