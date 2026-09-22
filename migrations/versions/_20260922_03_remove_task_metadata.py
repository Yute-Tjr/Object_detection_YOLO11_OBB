"""Remove task metadata fields.

Revision ID: 20260922_03
Revises: 20260920_02
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = "20260922_03"
down_revision: str | None = "20260920_02"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_column("inspection_tasks", "operator")
    op.drop_column("inspection_tasks", "name")
    op.drop_column("inspection_tasks", "note")


def downgrade() -> None:
    op.add_column(
        "inspection_tasks",
        sa.Column("operator", sa.String(length=128), nullable=True),
    )
    op.add_column(
        "inspection_tasks",
        sa.Column("name", sa.String(length=255), nullable=True),
    )
    op.add_column(
        "inspection_tasks",
        sa.Column("note", sa.Text(), nullable=True),
    )
