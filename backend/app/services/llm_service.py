from __future__ import annotations

import logging
import time
from typing import Any

from app.core.config import settings

logger = logging.getLogger(__name__)

DISCLAIMER = (
    "This report is AI-generated and intended for educational/assistive purposes only. "
    "It is not a medical diagnosis — consult a licensed radiologist or physician."
)

SYSTEM_PROMPT = (
    "You are an AI assistant that writes clear, structured, non-diagnostic "
    "explanations of chest X-ray model outputs for clinicians and patients. "
    "Always include a disclaimer that this is not a medical diagnosis. "
    "Do not invent patient history. Keep language cautious and non-speculative."
)


def build_user_prompt(label: str, confidence: float, region: str) -> str:
    return (
        f"Prediction: {label}\n"
        f"Confidence: {confidence:.4f}\n"
        f"Grad-CAM focus region: {region}\n\n"
        "Write a short report with: (1) Summary, (2) What the model observed, "
        "(3) Confidence interpretation, (4) Recommended next step, (5) Disclaimer."
    )


class LLMService:
    """Generate clinician-facing assistive reports from prediction context."""

    def generate_report(
        self,
        *,
        label: str,
        confidence: float,
        region_description: str,
    ) -> str:
        t0 = time.perf_counter()
        prompt = build_user_prompt(label, confidence, region_description)

        if not settings.llm_api_key or settings.llm_provider == "stub":
            report = self._template_report(label, confidence, region_description)
        else:
            try:
                report = self._call_openai(prompt)
            except Exception:
                logger.exception("LLM API failed; using template report")
                report = self._template_report(label, confidence, region_description)

        if DISCLAIMER not in report:
            report = f"{report.rstrip()}\n\n{DISCLAIMER}"

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info("LLM report generated ms=%.1f", elapsed_ms)
        return report

    def _call_openai(self, user_prompt: str) -> str:
        from openai import OpenAI

        client = OpenAI(api_key=settings.llm_api_key)
        response = client.chat.completions.create(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        return (response.choices[0].message.content or "").strip()

    @staticmethod
    def _template_report(label: str, confidence: float, region: str) -> str:
        return (
            f"1) Summary\n"
            f"The model classified this chest X-ray as **{label}** "
            f"with {confidence:.1%} confidence.\n\n"
            f"2) What the model observed\n"
            f"Grad-CAM highlights concentrated in the {region}. "
            f"These activation patterns informed the {label} prediction.\n\n"
            f"3) Confidence interpretation\n"
            f"A softmax confidence of {confidence:.1%} indicates "
            f"{'high' if confidence >= 0.85 else 'moderate' if confidence >= 0.65 else 'limited'} "
            f"model certainty for this image under the trained distribution.\n\n"
            f"4) Recommended next step\n"
            f"Correlate with clinical history and labs, and seek radiologist review "
            f"before any clinical decision.\n\n"
            f"5) Disclaimer\n"
            f"{DISCLAIMER}"
        )


_llm_singleton: LLMService | None = None


def get_llm_service() -> LLMService:
    global _llm_singleton
    if _llm_singleton is None:
        _llm_singleton = LLMService()
    return _llm_singleton
