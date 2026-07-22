"""Tests for app.llm.report_generator."""

from app.llm.report_generator import DISCLAIMER, _ensure_disclaimer, generate_report


def test_generate_report_stub_ends_with_disclaimer(monkeypatch):
    monkeypatch.setattr("app.llm.report_generator.settings.llm_api_key", "")
    monkeypatch.setattr("app.llm.report_generator.settings.llm_provider", "stub")

    report = generate_report("Pneumonia", 0.91)

    assert "Pneumonia" in report
    assert "91.0%" in report or "0.91" in report
    assert report.rstrip().endswith(DISCLAIMER)
    assert "not a medical diagnosis" in report.lower()
    assert "doctor" in report.lower()


def test_generate_report_normal_label(monkeypatch):
    monkeypatch.setattr("app.llm.report_generator.settings.llm_api_key", "")
    monkeypatch.setattr("app.llm.report_generator.settings.llm_provider", "stub")

    report = generate_report("Normal", 0.88, region_hint="mid-central lung field")
    assert "Normal" in report
    assert "mid-central" in report
    assert report.rstrip().endswith(DISCLAIMER)


def test_ensure_disclaimer_appends_when_missing():
    text = "Finding looks like pneumonia patterns."
    out = _ensure_disclaimer(text)
    assert out.endswith(DISCLAIMER)
    assert text in out


def test_ensure_disclaimer_dedupes():
    text = f"Summary here.\n\n{DISCLAIMER}"
    out = _ensure_disclaimer(text)
    assert out.count("Disclaimer:") == 1
