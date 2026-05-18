"""PMI-specific risk signal rules for Emigria inference."""

from __future__ import annotations

from typing import Any


HIGH_RISK_KEYWORDS = [
    "visa turis",
    "visa ziarah",
    "visa umroh",
    "tourist visa",
    "send passport",
    "kirim paspor",
    "bayar di muka",
    "dibayar di muka",
    "upfront payment",
    "advance payment",
    "biaya administrasi",
    "administrative fee",
    "langsung berangkat",
    "immediate departure",
    "proses cepat",
    "fast process",
    "tanpa potong gaji",
    "gaji tidak dipotong",
]

MEDIUM_RISK_KEYWORDS = [
    "gaji besar",
    "high salary",
    "penghasilan jutaan",
    "guaranteed income",
    "langsung diterima",
    "tanpa pengalaman",
    "no experience",
    "tanpa ijazah",
    "no degree",
    "hubungi sekarang",
    "contact now",
    "segera lamar",
    "apply immediately",
    "kuota terbatas",
    "limited slots",
    "urgent hiring",
]

FREE_EMAIL_DOMAINS = [
    "gmail.com",
    "yahoo.com",
    "hotmail.com",
    "outlook.com",
    "ymail.com",
    "aol.com",
    "protonmail.com",
]


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def build_search_text(record: dict[str, Any]) -> str:
    fields = [
        "title",
        "description",
        "requirements",
        "benefits",
        "company_profile",
        "salary_range",
    ]
    return " ".join(_as_text(record.get(field)) for field in fields).lower()


def calculate_pmi_rule_score(record: dict[str, Any]) -> tuple[int, list[str]]:
    """Return PMI rule score and triggered explainable signals."""
    text = build_search_text(record)
    score = 0
    triggered_rules: list[str] = []

    for keyword in HIGH_RISK_KEYWORDS:
        if keyword in text:
            score += 3
            triggered_rules.append(keyword)

    for keyword in MEDIUM_RISK_KEYWORDS:
        if keyword in text:
            score += 2
            triggered_rules.append(keyword)

    if not _as_text(record.get("company_profile")).strip():
        score += 2
        triggered_rules.append("company_profile_empty")

    try:
        has_company_logo = int(record.get("has_company_logo") or 0)
    except (TypeError, ValueError):
        has_company_logo = 0
    if has_company_logo == 0:
        score += 1
        triggered_rules.append("no_company_logo")

    try:
        has_questions = int(record.get("has_questions") or 0)
    except (TypeError, ValueError):
        has_questions = 0
    if has_questions == 0:
        score += 1
        triggered_rules.append("no_screening_questions")

    for domain in FREE_EMAIL_DOMAINS:
        if domain in text:
            score += 2
            triggered_rules.append(f"free_email:{domain}")
            break

    return score, triggered_rules


def combine_ml_and_pmi_rules(
    record: dict[str, Any],
    ml_probability: float,
    threshold: float,
    pmi_rule_threshold: int = 4,
) -> dict[str, Any]:
    pmi_rule_score, triggered_rules = calculate_pmi_rule_score(record)
    ml_prediction = int(ml_probability >= threshold)
    final_prediction = int(ml_prediction == 1 or pmi_rule_score >= pmi_rule_threshold)

    return {
        "ml_fraud_probability": float(ml_probability),
        "ml_fraud_prediction": ml_prediction,
        "pmi_rule_score": int(pmi_rule_score),
        "triggered_rules": triggered_rules,
        "fraud_prediction": final_prediction,
        "risk_level": "HIGH_RISK" if final_prediction else "LOW_RISK",
        "threshold": float(threshold),
        "pmi_rule_threshold": int(pmi_rule_threshold),
    }
