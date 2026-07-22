"""
LLM radiology-style report generator.

Primary entrypoint: ``generate_report(prediction_label, confidence_score)``.

Calls OpenAI or Gemini when ``LLM_API_KEY`` is set; otherwise uses a safe
template. Every returned string ends with a non-diagnosis disclaimer.
"""

from __future__ import annotations

import logging
import re

from app.core.config import settings

logger = logging.getLogger(__name__)

DISCLAIMER = (
    "Disclaimer: This is not a medical diagnosis. An AI model produced this "
    "assistive summary for educational purposes only. A qualified doctor or "
    "radiologist must review the image and confirm any clinical finding before "
    "treatment decisions are made."
)

SYSTEM_PROMPT = (
    "You are a medical writing assistant. Write a short, plain-language, "
    "radiology-style explanation of a chest X-ray AI model result for a "
    "non-specialist reader (patient or general clinician).\n\n"
    "Rules:\n"
    "- Use everyday words; avoid unexplained jargon.\n"
    "- Be cautious and non-speculative; do not invent symptoms or history.\n"
    "- Do not present the AI output as a confirmed diagnosis.\n"
    "- Respect the given confidence score; do not overstate certainty.\n"
    "- Keep the body under approximately 180 words.\n"
    "- You may omit a disclaimer in your draft; the application will append one."
)


def _user_prompt(
    prediction_label: str,
    confidence_score: float,
    region_hint: str | None = None,
) -> str:
    conf_pct = f"{confidence_score * 100:.1f}%"
    lines = [
        "Chest X-ray AI result",
        f"Predicted finding: {prediction_label}",
        f"Model confidence: {conf_pct} (softmax={confidence_score:.4f})",
    ]
    if region_hint:
        lines.append(f"Model attention hint (Grad-CAM): {region_hint}")
    lines.extend(
        [
            "",
            "Write a short radiology-style report in plain language that:",
            "1) States the AI finding in everyday words",
            "2) Briefly explains what that may mean on a chest X-ray",
            "3) Interprets the confidence level carefully",
            "4) Advises that a doctor must confirm before any clinical action",
        ]
    )
    return "\n".join(lines)


def _ensure_disclaimer(text: str) -> str:
    """Strip any trailing disclaimer variants, then append the canonical one."""
    body = (text or "").strip()
    if not body:
        return DISCLAIMER

    # Remove a trailing disclaimer-like paragraph so we don't duplicate.
    body = re.sub(
        r"(?:\n|\r\n)*Disclaimer:.*?(?:diagnosis|doctor|radiologist).*?\s*$",
        "",
        body,
        flags=re.IGNORECASE | re.DOTALL,
    ).rstrip()
    lower = body.lower()
    if "not a medical diagnosis" in lower and body.rstrip().endswith(DISCLAIMER):
        return body
    return f"{body}\n\n{DISCLAIMER}"


def _template_report(
    prediction_label: str,
    confidence_score: float,
    region_hint: str | None = None,
) -> str:
    """Offline / stub report when no LLM API key is configured."""
    if confidence_score >= 0.85:
        level = "high"
    elif confidence_score >= 0.65:
        level = "moderate"
    else:
        level = "limited"

    region_sentence = (
        f" The model's attention map highlighted the {region_hint}."
        if region_hint
        else ""
    )
    return (
        f"AI chest X-ray summary\n\n"
        f"The automated reading suggests a finding of \"{prediction_label}\" "
        f"with {confidence_score * 100:.1f}% model confidence ({level} certainty "
        f"for this image under the model's training data).{region_sentence}\n\n"
        f"In plain terms, the computer program saw patterns that, in its training "
        f"set, were more often linked to {prediction_label.lower()} than to the "
        f"other class it knows. Image quality, positioning, and conditions outside "
        f"the training set can all change this result.\n\n"
        f"Please share this image and summary with a doctor or radiologist so they "
        f"can confirm the finding before any treatment or further testing decisions."
    )


def _call_openai(system: str, user: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=settings.llm_api_key)
    response = client.chat.completions.create(
        model=settings.llm_model or "gpt-4o-mini",
        temperature=min(float(settings.llm_temperature), 0.3),
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return (response.choices[0].message.content or "").strip()


def _call_gemini(system: str, user: str) -> str:
    import google.generativeai as genai

    genai.configure(api_key=settings.llm_api_key)
    model_name = settings.llm_model or "gemini-1.5-flash"
    if model_name.startswith("gpt"):
        model_name = "gemini-1.5-flash"

    model = genai.GenerativeModel(
        model_name=model_name,
        system_instruction=system,
    )
    response = model.generate_content(
        user,
        generation_config={
            "temperature": min(float(settings.llm_temperature), 0.3),
        },
    )
    return (getattr(response, "text", None) or "").strip()


def generate_report(
    prediction_label: str,
    confidence_score: float,
    *,
    region_hint: str | None = None,
    provider: str | None = None,
) -> str:
    """
    Generate a short, plain-language radiology-style assistive report.

    Parameters
    ----------
    prediction_label:
        Model class label (e.g. ``\"Pneumonia\"`` or ``\"Normal\"``).
    confidence_score:
        Softmax probability in ``[0, 1]``.
    region_hint:
        Optional Grad-CAM region description.
    provider:
        Override ``settings.llm_provider``: ``openai`` | ``gemini`` | ``stub``.

    Returns
    -------
    str
        Report that **always ends** with a disclaimer that this is not a medical
        diagnosis and a doctor must confirm.
    """
    label = (prediction_label or "Unknown").strip()
    confidence = min(1.0, max(0.0, float(confidence_score)))
    provider_name = (provider or settings.llm_provider or "stub").strip().lower()
    user = _user_prompt(label, confidence, region_hint)

    if not settings.llm_api_key or provider_name in {"stub", "template", "none", ""}:
        logger.info("LLM provider=%s — using template report", provider_name or "stub")
        return _ensure_disclaimer(_template_report(label, confidence, region_hint))

    try:
        if provider_name in {"openai", "gpt"}:
            text = _call_openai(SYSTEM_PROMPT, user)
        elif provider_name in {"gemini", "google"}:
            text = _call_gemini(SYSTEM_PROMPT, user)
        else:
            logger.warning("Unknown LLM provider %r — trying OpenAI", provider_name)
            text = _call_openai(SYSTEM_PROMPT, user)
        return _ensure_disclaimer(text)
    except Exception:
        logger.exception(
            "LLM API call failed (provider=%s); using template report",
            provider_name,
        )
        return _ensure_disclaimer(_template_report(label, confidence, region_hint))
