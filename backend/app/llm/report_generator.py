"""
LLM radiology-style report generator.

Primary entrypoint: ``generate_report(...)``.

Accepts a **list** of detected conditions ``[{label, confidence}, ...]`` (and
still accepts a legacy single ``prediction_label`` + ``confidence_score``).

Calls OpenAI or Gemini when ``LLM_API_KEY`` is set; otherwise uses a safe
template. Every returned string ends with a non-diagnosis disclaimer.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Sequence

from app.core.config import settings

logger = logging.getLogger(__name__)

DISCLAIMER = (
    "Disclaimer: This is not a medical diagnosis. An AI model produced this "
    "assistive summary for educational purposes only. A licensed physician or "
    "radiologist must review the image and confirm any clinical finding before "
    "treatment decisions are made."
)

# Labels treated as "no disease finding" when alone
_NORMALISH = {"normal", "no tumor", "benign", "no finding", "clear"}

SYSTEM_PROMPT = (
    "You are a medical writing assistant. Write a short, plain-language, "
    "radiology-style explanation of an imaging AI model result for a "
    "non-specialist reader (patient or general clinician).\n\n"
    "Rules:\n"
    "- Use everyday words; avoid unexplained jargon.\n"
    "- Keep a calm, non-alarming tone. Never scare the reader.\n"
    "- Be cautious and non-speculative; do not invent symptoms or history.\n"
    "- Do not present the AI output as a confirmed diagnosis.\n"
    "- If several conditions are listed, address EACH one, note possible "
    "co-occurrence, and discuss them in confidence rank order (highest first).\n"
    "- If no conditions cleared the model threshold, say the scan appears "
    "normal or inconclusive for the conditions the model knows — not that "
    "the person is definitely healthy.\n"
    "- Respect given confidence scores; do not overstate certainty.\n"
    "- Keep the body under approximately 220 words.\n"
    "- You may omit a disclaimer in your draft; the application will append one."
)


def _as_condition_list(
    conditions: Sequence[Any] | None,
    prediction_label: str | None,
    confidence_score: float | None,
) -> list[dict[str, Any]]:
    """Normalize inputs to a ranked list of ``{label, confidence}`` dicts."""
    items: list[dict[str, Any]] = []

    if conditions:
        for item in conditions:
            if hasattr(item, "model_dump"):
                data = item.model_dump()
            elif isinstance(item, dict):
                data = item
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                data = {"label": item[0], "confidence": item[1]}
            else:
                continue
            label = str(data.get("label", "")).strip()
            if not label:
                continue
            try:
                conf = float(data.get("confidence", 0.0))
            except (TypeError, ValueError):
                conf = 0.0
            items.append(
                {
                    "label": label,
                    "confidence": min(1.0, max(0.0, conf)),
                }
            )

    if not items and prediction_label:
        conf = 0.0 if confidence_score is None else float(confidence_score)
        items.append(
            {
                "label": str(prediction_label).strip() or "Unknown",
                "confidence": min(1.0, max(0.0, conf)),
            }
        )

    items.sort(key=lambda x: x["confidence"], reverse=True)
    return items


def _is_empty_or_normal_only(conditions: list[dict[str, Any]]) -> bool:
    if not conditions:
        return True
    if len(conditions) == 1 and conditions[0]["label"].strip().lower() in _NORMALISH:
        return True
    return False


def _format_condition_lines(conditions: list[dict[str, Any]]) -> str:
    if not conditions:
        return "(none above the detection threshold)"
    lines = []
    for i, c in enumerate(conditions, start=1):
        lines.append(
            f"{i}. {c['label']} — {c['confidence'] * 100:.1f}% "
            f"(score={c['confidence']:.4f})"
        )
    return "\n".join(lines)


def _user_prompt(
    conditions: list[dict[str, Any]],
    *,
    scan_type: str | None = None,
    region_hint: str | None = None,
) -> str:
    modality = {
        "chest_xray": "Chest X-ray",
        "brain_mri": "Brain MRI",
        "skin_lesion": "Skin lesion image",
    }.get((scan_type or "chest_xray").lower(), "Medical image")

    empty = not conditions
    normal_only = _is_empty_or_normal_only(conditions) and not empty

    lines = [
        f"{modality} AI result",
        f"Detected conditions (ranked by confidence, highest first):",
        _format_condition_lines(conditions),
    ]
    if region_hint:
        lines.append(f"Model attention hint (Grad-CAM): {region_hint}")

    lines.append("")
    if empty:
        lines.extend(
            [
                "No condition cleared the model's confidence threshold.",
                "Write a short, calm, plain-language report that:",
                "1) States that the automated reading did not flag a clear finding "
                "above threshold (appears normal or inconclusive for the conditions "
                "this model was trained on)",
                "2) Explains that inconclusive does not guarantee perfect health",
                "3) Advises that a licensed physician must still review the image",
            ]
        )
    elif normal_only:
        label = conditions[0]["label"]
        conf = conditions[0]["confidence"]
        lines.extend(
            [
                f"The top result is \"{label}\" at {conf * 100:.1f}% confidence.",
                "Write a short, calm, plain-language report that:",
                "1) States the AI leans toward a normal / no-finding reading",
                "2) Interprets the confidence carefully without over-reassurance",
                "3) Advises that a licensed physician must confirm before any clinical action",
            ]
        )
    else:
        disease_items = [
            c for c in conditions if c["label"].strip().lower() not in _NORMALISH
        ]
        multi = len(disease_items) > 1
        lines.extend(
            [
                "Write a short, calm, plain-language radiology-style report that:",
                "1) Addresses EACH detected condition in confidence rank order",
                "2) Briefly explains what each may mean on this type of scan",
                (
                    "3) Notes that more than one finding can co-occur on the same image "
                    "and that co-occurrence is possible here"
                    if multi
                    else "3) Interprets the confidence of the main finding carefully"
                ),
                "4) Avoids alarming language; frames results as assistive, not definitive",
                "5) Advises that a licensed physician must confirm before any clinical action",
            ]
        )
    return "\n".join(lines)


def _ensure_disclaimer(text: str) -> str:
    """Strip any trailing disclaimer variants, then append the canonical one."""
    body = (text or "").strip()
    if not body:
        return DISCLAIMER

    body = re.sub(
        r"(?:\n|\r\n)*Disclaimer:.*?(?:diagnosis|doctor|radiologist|physician).*?\s*$",
        "",
        body,
        flags=re.IGNORECASE | re.DOTALL,
    ).rstrip()
    lower = body.lower()
    if "not a medical diagnosis" in lower and body.rstrip().endswith(DISCLAIMER):
        return body
    return f"{body}\n\n{DISCLAIMER}"


def _confidence_word(score: float) -> str:
    if score >= 0.85:
        return "high"
    if score >= 0.65:
        return "moderate"
    return "limited"


def _template_report(
    conditions: list[dict[str, Any]],
    *,
    scan_type: str | None = None,
    region_hint: str | None = None,
) -> str:
    """Offline / stub report when no LLM API key is configured."""
    modality = {
        "chest_xray": "chest X-ray",
        "brain_mri": "brain MRI",
        "skin_lesion": "skin lesion",
    }.get((scan_type or "chest_xray").lower(), "imaging")

    region_sentence = (
        f" The model's attention map highlighted the {region_hint}."
        if region_hint
        else ""
    )

    if not conditions:
        return (
            f"AI {modality} summary\n\n"
            f"The automated reading did not flag any of the conditions it looks "
            f"for above the usual confidence threshold. In plain terms, the scan "
            f"looks normal or inconclusive for those trained labels — not a "
            f"guarantee that everything is fine.{region_sentence}\n\n"
            f"Image quality, positioning, and conditions outside the training set "
            f"can all change this result. Please share the image with a licensed "
            f"physician or radiologist so they can review it before any clinical "
            f"decisions."
        )

    if _is_empty_or_normal_only(conditions):
        c = conditions[0]
        return (
            f"AI {modality} summary\n\n"
            f"The automated reading leans toward \"{c['label']}\" with "
            f"{c['confidence'] * 100:.1f}% model confidence "
            f"({_confidence_word(c['confidence'])} certainty for this image)."
            f"{region_sentence}\n\n"
            f"In plain terms, the program did not strongly match disease patterns "
            f"it was trained to notice. That is reassuring but not a final all-clear. "
            f"Please share this image and summary with a licensed physician so they "
            f"can confirm the finding before any further decisions."
        )

    disease = [c for c in conditions if c["label"].strip().lower() not in _NORMALISH]
    ranked = disease or conditions
    bullets = []
    for c in ranked:
        bullets.append(
            f"- {c['label']}: {c['confidence'] * 100:.1f}% "
            f"({_confidence_word(c['confidence'])} model confidence)"
        )
    ranking_block = "\n".join(bullets)

    if len(ranked) == 1:
        c = ranked[0]
        body = (
            f"The automated reading suggests a finding of \"{c['label']}\" "
            f"with {c['confidence'] * 100:.1f}% model confidence "
            f"({_confidence_word(c['confidence'])} certainty for this image)."
            f"{region_sentence}\n\n"
            f"In plain terms, the computer program saw patterns that, in its "
            f"training set, were more often linked to {c['label'].lower()} than "
            f"to other labels it knows."
        )
    else:
        names = ", ".join(c["label"] for c in ranked[:-1]) + f", and {ranked[-1]['label']}"
        body = (
            f"The automated reading flagged more than one possible finding on "
            f"this {modality}. Ranked by model confidence (highest first):\n"
            f"{ranking_block}\n\n"
            f"These patterns can co-occur on the same image; seeing {names} "
            f"together does not by itself prove severity or rule anything in or "
            f"out.{region_sentence}\n\n"
            f"In plain terms, the program matched several training patterns at "
            f"once. That is a reason for careful clinical review, not for alarm."
        )

    return (
        f"AI {modality} summary\n\n"
        f"{body}\n\n"
        f"Image quality, positioning, and conditions outside the training set "
        f"can all change this result. Please share this image and summary with a "
        f"licensed physician or radiologist so they can confirm any finding "
        f"before treatment or further testing decisions."
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
    prediction_label: str | None = None,
    confidence_score: float | None = None,
    *,
    conditions: Sequence[Any] | None = None,
    scan_type: str | None = None,
    region_hint: str | None = None,
    provider: str | None = None,
) -> str:
    """
    Generate a short, plain-language radiology-style assistive report.

    Parameters
    ----------
    conditions:
        Preferred input — list of ``{label, confidence}`` (or Pydantic models).
        Ranked by confidence descending inside this function.
    prediction_label / confidence_score:
        Legacy single-finding inputs (still supported).
    scan_type:
        Optional modality hint (``chest_xray``, ``brain_mri``, ``skin_lesion``).
    region_hint:
        Optional Grad-CAM region description.
    provider:
        Override ``settings.llm_provider``: ``openai`` | ``gemini`` | ``stub``.

    Returns
    -------
    str
        Report that **always ends** with a disclaimer that this is not a medical
        diagnosis and a licensed physician must confirm.
    """
    ranked = _as_condition_list(conditions, prediction_label, confidence_score)
    provider_name = (provider or settings.llm_provider or "stub").strip().lower()
    user = _user_prompt(ranked, scan_type=scan_type, region_hint=region_hint)

    if not settings.llm_api_key or provider_name in {"stub", "template", "none", ""}:
        logger.info("LLM provider=%s — using template report", provider_name or "stub")
        return _ensure_disclaimer(
            _template_report(ranked, scan_type=scan_type, region_hint=region_hint)
        )

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
        return _ensure_disclaimer(
            _template_report(ranked, scan_type=scan_type, region_hint=region_hint)
        )
