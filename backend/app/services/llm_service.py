from __future__ import annotations

import logging
import time

from app.llm.report_generator import generate_report as llm_generate_report

logger = logging.getLogger(__name__)


class LLMService:
    """Thin service wrapper around ``app.llm.report_generator``."""

    def generate_report(
        self,
        *,
        label: str,
        confidence: float,
        region_description: str = "",
    ) -> str:
        t0 = time.perf_counter()
        report = llm_generate_report(
            prediction_label=label,
            confidence_score=confidence,
            region_hint=region_description or None,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info("LLM report generated ms=%.1f", elapsed_ms)
        return report


_llm_singleton: LLMService | None = None


def get_llm_service() -> LLMService:
    global _llm_singleton
    if _llm_singleton is None:
        _llm_singleton = LLMService()
    return _llm_singleton
