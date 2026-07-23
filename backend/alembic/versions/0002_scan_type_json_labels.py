"""Add scan_type; store prediction_label as JSON list of {label, confidence}.

Revision ID: 0002_scan_type_json_labels
Revises: 0001_initial
Create Date: 2026-07-23
"""

from __future__ import annotations

import json
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0002_scan_type_json_labels"
down_revision: Union[str, None] = "0001_initial"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()

    with op.batch_alter_table("predictions") as batch_op:
        batch_op.add_column(sa.Column("scan_type", sa.String(length=32), nullable=True))

    # Backfill modality for existing rows
    bind.execute(sa.text("UPDATE predictions SET scan_type = 'chest_xray' WHERE scan_type IS NULL"))

    # Convert plain-string prediction_label values → JSON list
    rows = bind.execute(
        sa.text("SELECT id, prediction_label, confidence FROM predictions")
    ).fetchall()
    for row in rows:
        raw = row.prediction_label if hasattr(row, "prediction_label") else row[1]
        conf = row.confidence if hasattr(row, "confidence") else row[2]
        row_id = row.id if hasattr(row, "id") else row[0]
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                continue
        except (TypeError, json.JSONDecodeError):
            pass
        payload = json.dumps(
            [{"label": str(raw), "confidence": float(conf)}],
            separators=(",", ":"),
        )
        bind.execute(
            sa.text("UPDATE predictions SET prediction_label = :payload WHERE id = :id"),
            {"payload": payload, "id": row_id},
        )

    with op.batch_alter_table("predictions") as batch_op:
        batch_op.alter_column(
            "prediction_label",
            existing_type=sa.String(length=50),
            type_=sa.Text(),
            nullable=False,
        )
        batch_op.alter_column(
            "scan_type",
            existing_type=sa.String(length=32),
            nullable=False,
        )
        batch_op.create_index("ix_predictions_scan_type", ["scan_type"])


def downgrade() -> None:
    bind = op.get_bind()

    # Best-effort: collapse JSON list back to primary label string
    rows = bind.execute(sa.text("SELECT id, prediction_label FROM predictions")).fetchall()
    for row in rows:
        raw = row.prediction_label if hasattr(row, "prediction_label") else row[1]
        row_id = row.id if hasattr(row, "id") else row[0]
        label = str(raw)
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list) and parsed:
                label = str(parsed[0].get("label", label))[:50]
        except (TypeError, json.JSONDecodeError):
            label = label[:50]
        bind.execute(
            sa.text("UPDATE predictions SET prediction_label = :label WHERE id = :id"),
            {"label": label, "id": row_id},
        )

    with op.batch_alter_table("predictions") as batch_op:
        batch_op.drop_index("ix_predictions_scan_type")
        batch_op.alter_column(
            "prediction_label",
            existing_type=sa.Text(),
            type_=sa.String(length=50),
            nullable=False,
        )
        batch_op.drop_column("scan_type")
