"""predictions table matching app.db.models.Prediction

Revision ID: 0001_initial
Revises:
Create Date: 2026-07-22
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0001_initial"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "predictions",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("image_path", sa.String(length=512), nullable=False),
        sa.Column("heatmap_path", sa.String(length=512), nullable=False),
        sa.Column("prediction_label", sa.String(length=50), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("report_text", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_predictions_id", "predictions", ["id"])
    op.create_index("ix_predictions_created_at", "predictions", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_predictions_created_at", table_name="predictions")
    op.drop_index("ix_predictions_id", table_name="predictions")
    op.drop_table("predictions")
