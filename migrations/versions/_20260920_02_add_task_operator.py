"""Add operator metadata to inspection tasks.

Revision ID: 20260920_02
Revises: 20260920_01
Create Date: 2026-09-20
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "20260920_02"
down_revision: str | None = "20260920_01"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "inspection_tasks",
        sa.Column("operator", sa.String(length=128), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("inspection_tasks", "operator")
