"""Add source metadata to image feedback items.

Revision ID: 20260923_05
Revises: 20260923_04
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = "20260923_05"
down_revision: str | None = "20260923_04"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "image_feedback_items",
        sa.Column("source", sa.String(length=16), nullable=True),
    )
    op.execute(
        sa.text("UPDATE image_feedback_items SET source = 'manual' WHERE source IS NULL")
    )
    with op.batch_alter_table("image_feedback_items") as batch_op:
        batch_op.alter_column(
            "source",
            existing_type=sa.String(length=16),
            nullable=False,
        )
        batch_op.alter_column(
            "verdict",
            existing_type=sa.String(length=2),
            nullable=True,
        )
        batch_op.create_check_constraint(
            "ck_feedback_item_source",
            "source IN ('manual', 'model', 'unreviewed')",
        )


def downgrade() -> None:
    connection = op.get_bind()
    null_verdicts = connection.scalar(
        sa.text("SELECT count(*) FROM image_feedback_items WHERE verdict IS NULL")
    )
    if null_verdicts:
        raise RuntimeError(
            "cannot downgrade while image feedback items have null verdicts"
        )
    with op.batch_alter_table("image_feedback_items") as batch_op:
        batch_op.drop_constraint("ck_feedback_item_source", type_="check")
        batch_op.alter_column(
            "verdict",
            existing_type=sa.String(length=2),
            nullable=False,
        )
        batch_op.drop_column("source")
