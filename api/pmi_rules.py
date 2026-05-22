"""PMI-specific risk signal rules for Emigria hybrid inference.

This layer keeps the MLP as the base fraud model, then adds a PMI domain
adaptation score for Indonesian migrant-worker job offers.
"""

from __future__ import annotations

from typing import Any


ML_WEIGHT = 0.3
PMI_WEIGHT = 0.7
PMI_SCORE_CAP = 10
REVIEW_THRESHOLD = 0.40
HIGH_RISK_THRESHOLD = 0.60


HIGH_RISK_KEYWORDS = [
    "visa turis",
    "visa wisata",
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
    "biaya keberangkatan",
    "uang muka",
    "dp dulu",
    "biaya admin",
    "langsung berangkat",
    "immediate departure",
    "berangkat hari ini",
    "proses keberangkatan cepat",
    "proses cepat",
    "fast process",
    "proses kilat",
    "tanpa potong gaji",
    "gaji tidak dipotong",
    "kontrak tidak jelas",
    "tanpa kontrak",
    "kontrak tidak transparan",
    "jam kerja berlebihan",
    "12 jam kerja",
    "14 jam kerja",
    "16 jam kerja",
    "tidak boleh keluar",
    "dilarang keluar",
    "mess tertutup",
    "tinggal di mess",
    "paspor disimpan",
    "paspor ditahan",
    "paspor dipegang agen",
    "menyerahkan paspor",
    "paspor asli",
    "dokumen asli",
    "data sensitif",
    "kode otp",
    "pin atm",
    "foto ktp selfie",
    "visa on arrival",
    "izin operasional dalam proses",
    "legalitas dalam proses",
    "izin sedang diurus",
    "tanpa prosedur resmi",
    "tanpa tes resmi",
    "casino online",
    "online gambling",
    "judi online",
    "scam center",
    "live streaming ilegal",
    "operator situs",
    "game online",
    "macau",
    "kamboja",
    "cambodia",
    "myanmar",
    "laos",
]

MEDIUM_RISK_KEYWORDS = [
    "gaji besar",
    "high salary",
    "penghasilan jutaan",
    "gaji puluhan juta",
    "gaji tinggi",
    "gaji fantastis",
    "gaji tidak masuk akal",
    "guaranteed income",
    "bonus besar",
    "bonus harian",
    "bonus target",
    "berdasarkan target",
    "fasilitas mewah",
    "apartemen gratis",
    "akomodasi mewah",
    "langsung diterima",
    "tanpa seleksi",
    "tanpa interview",
    "interview mudah",
    "tanpa pengalaman",
    "tidak perlu pengalaman",
    "tidak memerlukan pengalaman",
    "no experience",
    "tanpa ijazah",
    "tidak perlu pendidikan",
    "no degree",
    "tanpa skill",
    "tanpa keahlian",
    "pendidikan minimal rendah",
    "hubungi sekarang",
    "contact now",
    "segera lamar",
    "apply immediately",
    "kuota terbatas",
    "limited slots",
    "urgent hiring",
    "wa.me",
    "t.me",
    "wa kami",
    "chat wa",
    "dm kami",
    "telegram pribadi",
    "wa pribadi",
    "nomor pribadi",
    "alamat tidak jelas",
    "alamat kantor tidak jelas",
    "tanpa alamat kantor",
    "format tidak profesional",
    "banyak typo",
    "tidak menjelaskan detail pekerjaan",
    "detail pekerjaan tidak jelas",
]

# Critical keywords. If two or more appear together, force HIGH_RISK as a
# safety net for obvious non-procedural placement red flags.
HARD_STOP_KEYWORDS = [
    "visa turis",
    "visa wisata",
    "tourist visa",
    "visa on arrival",
    "kirim paspor",
    "send passport",
    "paspor ditahan",
    "bayar di muka",
    "upfront payment",
    "advance payment",
    "biaya keberangkatan",
    "uang muka",
    "biaya admin",
    "biaya administrasi",
    "transfer uang",
    "membayar biaya",
    "biaya di awal",
    "menyerahkan paspor",
    "paspor disimpan",
    "paspor asli",
    "dokumen asli",
    "izin operasional dalam proses",
    "legalitas dalam proses",
    "tanpa prosedur resmi",
    "tanpa tes resmi",
    "scam center",
    "casino online",
    "judi online",
    "mess tertutup",
    "dilarang keluar",
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

# Legit signals that reduce the PMI rule score for official-looking postings.
LEGIT_SIGNAL_KEYWORDS = [
    "bp2mi",
    "bnp2tki",
    "p3mi",
    "sipmi",
    "sisnaker",
    "sipp tkis",
    "izin operasional",
    "berizin kemnaker",
    "terdaftar di kemnaker",
    "id.bp2mi.go.id",
    "kemnaker.go.id",
    "kontrak kerja resmi",
    "perjanjian penempatan",
    "tidak ada biaya penempatan",
]


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _normalize_text(text: str) -> str:
    lowered = text.lower()
    replacements = {
        "administrasi": "admin",
        "whats app": "whatsapp",
        "whatsapp": "wa",
        "brangkat": "berangkat",
        "20jt": "20 juta",
        "10jt": "10 juta",
        "5jt": "5 juta",
    }
    for source, target in replacements.items():
        lowered = lowered.replace(source, target)
    return " ".join(lowered.split())


def _percentage(value: float) -> float:
    return round(float(value) * 100, 2)


def build_search_text(record: dict[str, Any]) -> str:
    fields = [
        "title",
        "description",
        "requirements",
        "benefits",
        "company_profile",
        "salary_range",
    ]
    text = " ".join(_as_text(record.get(field)) for field in fields)
    return _normalize_text(text)


def calculate_pmi_rule_score(record: dict[str, Any]) -> tuple[int, list[str], int]:
    """Return PMI rule score, triggered explainable signals, and hard-stop count.

    The hard-stop count is used as a safety logic in the hybrid combiner.
    """
    text = build_search_text(record)
    score = 0
    triggered_rules: list[str] = []

    for keyword in HIGH_RISK_KEYWORDS:
        normalized_keyword = _normalize_text(keyword)
        if normalized_keyword in text:
            score += 3
            triggered_rules.append(keyword)

    for keyword in MEDIUM_RISK_KEYWORDS:
        normalized_keyword = _normalize_text(keyword)
        if normalized_keyword in text:
            score += 2
            triggered_rules.append(keyword)

    if not _as_text(record.get("company_profile")).strip():
        score += 1
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

    # Reduce the rule score if the posting has strong official PMI signals.
    legit_signal_count = 0
    for signal in LEGIT_SIGNAL_KEYWORDS:
        normalized_signal = _normalize_text(signal)
        if normalized_signal == "izin operasional" and "izin operasional dalam proses" in text:
            continue
        if normalized_signal in text:
            legit_signal_count += 1
    if legit_signal_count > 0:
        legit_reduction = min(legit_signal_count * 2, 4)  # max -4
        score = max(0, score - legit_reduction)
        triggered_rules.append(f"legit_signal_count:{legit_signal_count}")

    # Count critical red flags for the safety override.
    hard_stop_count = sum(
        1 for keyword in HARD_STOP_KEYWORDS if _normalize_text(keyword) in text
    )

    return score, triggered_rules, hard_stop_count


def combine_ml_and_pmi_rules(
    record: dict[str, Any],
    ml_probability: float,
    threshold: float,
    pmi_rule_threshold: int = 6,
) -> dict[str, Any]:
    """Combine MLP probability and PMI rules into one calibrated risk score."""
    pmi_rule_score, triggered_rules, hard_stop_count = calculate_pmi_rule_score(record)

    ml_score = max(0.0, min(float(ml_probability), 1.0))
    pmi_normalized_score = min(float(pmi_rule_score), float(PMI_SCORE_CAP)) / PMI_SCORE_CAP
    weighted_score = (ML_WEIGHT * ml_score) + (PMI_WEIGHT * pmi_normalized_score)

    ml_prediction = int(ml_probability >= threshold)
    rule_prediction = int(pmi_rule_score >= pmi_rule_threshold)
    hard_stop_triggered = int(hard_stop_count >= 2)

    final_risk_score = weighted_score
    if ml_prediction == 1 or hard_stop_triggered == 1:
        final_risk_score = max(final_risk_score, HIGH_RISK_THRESHOLD)

    if final_risk_score >= HIGH_RISK_THRESHOLD:
        risk_level = "HIGH_RISK"
    elif final_risk_score >= REVIEW_THRESHOLD:
        risk_level = "REVIEW"
    else:
        risk_level = "LOW_RISK"

    final_prediction = int(risk_level != "LOW_RISK")

    return {
        "ml_fraud_probability": float(ml_probability),
        "ml_fraud_percentage": _percentage(ml_score),
        "ml_fraud_prediction": ml_prediction,
        "pmi_rule_score": int(pmi_rule_score),
        "pmi_normalized_score": float(round(pmi_normalized_score, 4)),
        "pmi_risk_percentage": _percentage(pmi_normalized_score),
        "pmi_rule_prediction": rule_prediction,
        "triggered_rules": triggered_rules,
        "hard_stop_triggered": bool(hard_stop_triggered),
        "hard_stop_count": int(hard_stop_count),
        "ml_weight": ML_WEIGHT,
        "pmi_weight": PMI_WEIGHT,
        "final_risk_score": float(round(final_risk_score, 4)),
        "final_risk_percentage": _percentage(final_risk_score),
        "fraud_prediction": final_prediction,
        "risk_level": risk_level,
        "threshold": float(threshold),
        "pmi_rule_threshold": int(pmi_rule_threshold),
        "review_threshold": REVIEW_THRESHOLD,
        "high_risk_threshold": HIGH_RISK_THRESHOLD,
    }
