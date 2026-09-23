"""Add authentication sessions and image feedback.

Revision ID: 20260923_04
Revises: 20260922_03
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = "20260923_04"
down_revision: str | None = "20260922_03"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("username", sa.String(length=64), nullable=False),
        sa.Column("password_hash", sa.Text(), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("username"),
    )
    op.create_table(
        "user_sessions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("token_hash", sa.String(length=64), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("token_hash"),
    )
    op.create_index("ix_user_sessions_user_id", "user_sessions", ["user_id"])
    op.create_table(
        "image_feedbacks",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("image_id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["image_id"], ["inspection_images.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id", "image_id", name="uq_feedback_user_image"),
    )
    op.create_index("ix_image_feedbacks_image_id", "image_feedbacks", ["image_id"])
    op.create_index("ix_image_feedbacks_user_id", "image_feedbacks", ["user_id"])
    op.create_table(
        "image_feedback_items",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("feedback_id", sa.Uuid(), nullable=False),
        sa.Column("detection_id", sa.Uuid(), nullable=False),
        sa.Column("region_label", sa.String(length=64), nullable=False),
        sa.Column("verdict", sa.String(length=2), nullable=False),
        sa.Column("color", sa.String(length=1), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["feedback_id"], ["image_feedbacks.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(["detection_id"], ["detections.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "feedback_id", "detection_id", name="uq_feedback_detection"
        ),
    )
    op.create_index(
        "ix_image_feedback_items_detection_id",
        "image_feedback_items",
        ["detection_id"],
    )
    op.create_index(
        "ix_image_feedback_items_feedback_id",
        "image_feedback_items",
        ["feedback_id"],
    )
    op.create_table(
        "image_feedback_misses",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("feedback_id", sa.Uuid(), nullable=False),
        sa.Column("logical_region", sa.String(length=16), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["feedback_id"], ["image_feedbacks.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "feedback_id", "logical_region", name="uq_feedback_missed_region"
        ),
    )
    op.create_index(
        "ix_image_feedback_misses_feedback_id",
        "image_feedback_misses",
        ["feedback_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_image_feedback_misses_feedback_id", table_name="image_feedback_misses"
    )
    op.drop_table("image_feedback_misses")
    op.drop_index(
        "ix_image_feedback_items_feedback_id", table_name="image_feedback_items"
    )
    op.drop_index(
        "ix_image_feedback_items_detection_id", table_name="image_feedback_items"
    )
    op.drop_table("image_feedback_items")
    op.drop_index("ix_image_feedbacks_user_id", table_name="image_feedbacks")
    op.drop_index("ix_image_feedbacks_image_id", table_name="image_feedbacks")
    op.drop_table("image_feedbacks")
    op.drop_index("ix_user_sessions_user_id", table_name="user_sessions")
    op.drop_table("user_sessions")
    op.drop_table("users")
