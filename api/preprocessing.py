"""Preprocessing utilities mirroring the Emigria DS training pipeline."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd


TARGET_COL = "fraudulent"

RAW_COLUMNS = [
    "title",
    "location",
    "salary_range",
    "description",
    "requirements",
    "company_profile",
    "employment_type",
    "industry",
    "benefits",
    "telecommuting",
    "has_company_logo",
    "has_questions",
    "fraudulent",
    "required_experience",
    "required_education",
]

TEXT_DEFAULT_EMPTY = ["description", "requirements", "company_profile", "benefits"]
CATEGORY_DEFAULT_UNKNOWN = ["employment_type", "industry", "title"]
CATEGORY_DEFAULT_NOT_SPECIFIED = ["required_experience", "required_education"]
BINARY_COLUMNS = ["telecommuting", "has_company_logo", "has_questions"]
TARGET_ENCODE_COLS = ["country", "employment_type", "industry"]

TFIDF_MAX_FEATURES = 100
TFIDF_FEATURES = [f"tfidf_{i}" for i in range(TFIDF_MAX_FEATURES)]

BASE_FEATURES = [
    "salary_mid",
    "salary_spread",
    "salary_spread_ratio",
    "has_salary",
    "title_length",
    "desc_length",
    "req_length",
    "has_company_profile",
    "scam_keyword_score",
    "has_email_in_desc",
    "is_free_email",
    "exclamation_count",
    "is_indonesian_posting",
    "country_fraud_rate",
    "country_safety_score",
    "country_avg_salary",
    "salary_vs_country_avg",
    "telecommuting",
    "has_company_logo",
    "has_questions",
    "emp_type_fraud_rate",
    "industry_fraud_rate",
]

FEATURE_COLUMNS = BASE_FEATURES + TFIDF_FEATURES

COUNTRY_MAP = {
    "US": "United States",
    "GB": "United Kingdom",
    "UK": "United Kingdom",
    "DE": "Germany",
    "CA": "Canada",
    "AU": "Australia",
    "IN": "India",
    "NZ": "New Zealand",
    "SG": "Singapore",
    "MY": "Malaysia",
    "HK": "Hong Kong",
    "AE": "UAE",
    "SA": "Saudi Arabia",
    "JP": "Japan",
    "KR": "South Korea",
    "TW": "Taiwan",
    "FR": "France",
    "IE": "Ireland",
    "NL": "Netherlands",
    "CH": "Switzerland",
    "IT": "Italy",
    "ES": "Spain",
    "SE": "Sweden",
    "PL": "Poland",
    "IL": "Israel",
    "PH": "Philippines",
    "TH": "Thailand",
    "ID": "Indonesia",
    "BR": "Brazil",
    "ZA": "South Africa",
    "NG": "Nigeria",
    "QA": "Qatar",
    "KW": "Kuwait",
    "BH": "Bahrain",
    "JO": "Jordan",
    "CN": "China",
    "BN": "Brunei",
    "TR": "Turkey",
    "RU": "Russia",
    "RO": "Romania",
    "GR": "Greece",
    "BE": "Belgium",
    "AT": "Austria",
    "NO": "Norway",
    "DK": "Denmark",
    "FI": "Finland",
    "PK": "Pakistan",
    "BD": "Bangladesh",
    "EG": "Egypt",
    "OM": "Oman",
    "LB": "Lebanon",
    "MX": "Mexico",
    "AR": "Argentina",
    "CO": "Colombia",
    "CL": "Chile",
    "HR": "Croatia",
    "HU": "Hungary",
    "CZ": "Czech Republic",
    "PT": "Portugal",
    "UA": "Ukraine",
    "RS": "Serbia",
    "BG": "Bulgaria",
    "LK": "Sri Lanka",
    "MM": "Myanmar",
    "VN": "Vietnam",
}

SCAM_KW_HIGH = [
    "tanpa potong gaji",
    "visa turis",
    "visa ziarah",
    "visa umroh",
    "kirim paspor",
    "bayar di muka",
    "biaya administrasi",
    "proses cepat",
    "p3mi",
    "pjtki",
    "tidak dipungut biaya",
    "gratis visa",
    "gaji tidak dipotong",
    "agen resmi terpercaya",
    "guaranteed income",
    "no fee required",
    "free visa",
    "send passport",
    "upfront payment",
    "advance payment",
]

SCAM_KW_MED = [
    "kerja luar negeri",
    "gaji besar",
    "penghasilan jutaan",
    "langsung diterima",
    "tanpa pengalaman",
    "tanpa ijazah",
    "tidak perlu pengalaman",
    "work from home",
    "no experience needed",
    "no degree required",
    "urgent hiring",
    "immediate start",
    "earn money fast",
    "make money online",
    "high income",
]

SCAM_KW_LOW = [
    "hubungi sekarang",
    "daftar sekarang",
    "segera lamar",
    "terbatas",
    "kuota terbatas",
    "click here",
    "act now",
    "contact now",
    "apply immediately",
    "limited slots",
]

ID_WORDS = [
    "kami",
    "kerja",
    "gaji",
    "lowongan",
    "dibutuhkan",
    "segera",
    "lamar",
    "hubungi",
    "pria",
    "wanita",
]


def parse_salary(value: Any) -> tuple[float, float]:
    if pd.isna(value):
        return np.nan, np.nan

    text = str(value).strip()
    if text.lower() in ["not disclosed", "", "nan"]:
        return np.nan, np.nan

    text = text.replace("$", "").replace(",", "")

    if "k" in text.lower():
        parts = re.findall(r"([\d.]+)\s*k", text, re.IGNORECASE)
        if len(parts) >= 2:
            return float(parts[0]) * 1000, float(parts[1]) * 1000

    parts = re.findall(r"[\d.]+", text)
    if len(parts) >= 2:
        minimum, maximum = float(parts[0]), float(parts[1])
        if minimum < 500 and maximum < 500:
            minimum, maximum = minimum * 2080, maximum * 2080
        return minimum, maximum

    return np.nan, np.nan


def get_country_from_location(location: Any) -> str:
    if pd.isna(location):
        return "Unknown"

    text = str(location).strip()
    if not text:
        return "Unknown"

    code = text.split(",")[0].strip()
    return COUNTRY_MAP.get(code, code)


def choose_country(row: pd.Series) -> str:
    country = row.get("country", "")
    if pd.notna(country) and str(country).strip():
        return str(country).strip()
    return get_country_from_location(row.get("location", ""))


def calc_scam_score(text: Any) -> int:
    if not isinstance(text, str) or not text:
        return 0

    lowered = text.lower()
    score = 0
    score += sum(3 for keyword in SCAM_KW_HIGH if keyword in lowered)
    score += sum(2 for keyword in SCAM_KW_MED if keyword in lowered)
    score += sum(1 for keyword in SCAM_KW_LOW if keyword in lowered)
    return score


def prepare_base_dataframe(raw_df: pd.DataFrame, has_target: bool = False) -> pd.DataFrame:
    df = raw_df.copy()

    needed_cols = RAW_COLUMNS.copy()
    if not has_target:
        needed_cols = [col for col in needed_cols if col != TARGET_COL]

    for col in needed_cols:
        if col not in df.columns:
            df[col] = np.nan

    if "country" not in df.columns:
        df["country"] = np.nan

    df = df[needed_cols + ["country"]].copy()
    df["country"] = df.apply(choose_country, axis=1)

    parsed_salary = df["salary_range"].apply(parse_salary)
    df["salary_min"] = parsed_salary.apply(lambda values: values[0])
    df["salary_max"] = parsed_salary.apply(lambda values: values[1])

    for col in TEXT_DEFAULT_EMPTY:
        df[col] = df[col].fillna("").astype(str)

    for col in CATEGORY_DEFAULT_UNKNOWN:
        df[col] = df[col].fillna("Unknown").astype(str)
        df.loc[df[col].str.strip() == "", col] = "Unknown"

    for col in CATEGORY_DEFAULT_NOT_SPECIFIED:
        df[col] = df[col].fillna("Not Specified").astype(str)
        df.loc[df[col].str.strip() == "", col] = "Not Specified"

    df["location"] = df["location"].fillna("").astype(str)
    df["salary_range"] = df["salary_range"].fillna("").astype(str)
    df["country"] = df["country"].fillna("Unknown").astype(str)
    df.loc[df["country"].str.strip() == "", "country"] = "Unknown"

    for col in BINARY_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    if has_target:
        df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
        df = df.dropna(subset=[TARGET_COL])
        df[TARGET_COL] = df[TARGET_COL].astype(int)

    df["salary_mid"] = (df["salary_min"] + df["salary_max"]) / 2
    df["salary_spread"] = df["salary_max"] - df["salary_min"]
    df["salary_spread_ratio"] = np.where(
        df["salary_mid"] > 0,
        df["salary_spread"] / df["salary_mid"],
        np.nan,
    )
    df["has_salary"] = df["salary_min"].notna().astype(int)

    df["title_length"] = df["title"].str.len()
    df["desc_length"] = df["description"].str.len()
    df["req_length"] = df["requirements"].str.len()
    df["has_company_profile"] = (df["company_profile"].str.len() > 0).astype(int)

    df["scam_keyword_score"] = df["description"].apply(calc_scam_score)
    df["has_email_in_desc"] = df["description"].str.contains(
        r"[\w.-]+@[\w.-]+\.\w+",
        regex=True,
        na=False,
    ).astype(int)
    df["is_free_email"] = df["description"].str.contains(
        r"@(?:gmail|yahoo|hotmail|outlook|ymail|aol|mail\.ru|protonmail)\.\w+",
        regex=True,
        case=False,
        na=False,
    ).astype(int)
    df["exclamation_count"] = df["description"].str.count("!").fillna(0)
    df["is_indonesian_posting"] = df["description"].apply(
        lambda value: int(sum(word in str(value).lower() for word in ID_WORDS) >= 3)
    )

    return df


def transform_features(
    base_df: pd.DataFrame,
    target_encoder: Any,
    tfidf_vectorizer: Any,
    scaler: Any,
    country_salary_data: dict[str, Any],
    feature_columns: list[str] | None = None,
) -> np.ndarray:
    feature_columns = feature_columns or FEATURE_COLUMNS
    x_out = base_df.copy()

    encoded = target_encoder.transform(x_out[TARGET_ENCODE_COLS])
    x_out["country_fraud_rate"] = encoded["country"]
    x_out["country_safety_score"] = 1 - encoded["country"]
    x_out["emp_type_fraud_rate"] = encoded["employment_type"]
    x_out["industry_fraud_rate"] = encoded["industry"]

    country_salary_avg = country_salary_data["country_salary_avg"]
    global_salary_median = country_salary_data["global_salary_median"]

    x_out["country_avg_salary"] = x_out["country"].map(country_salary_avg).fillna(global_salary_median)
    x_out["salary_vs_country_avg"] = np.where(
        x_out["salary_mid"].notna() & (x_out["country_avg_salary"] > 0),
        x_out["salary_mid"] / x_out["country_avg_salary"],
        np.nan,
    )

    salary_cols = [
        "salary_mid",
        "salary_spread",
        "salary_spread_ratio",
        "salary_vs_country_avg",
    ]
    x_out[salary_cols] = x_out[salary_cols].fillna(-1)
    x_out["country_avg_salary"] = x_out["country_avg_salary"].fillna(global_salary_median)

    tfidf_matrix = tfidf_vectorizer.transform(x_out["description"].fillna("")).toarray()
    tfidf_df = pd.DataFrame(tfidf_matrix, columns=TFIDF_FEATURES, index=x_out.index)
    x_out = pd.concat([x_out, tfidf_df], axis=1)

    for col in feature_columns:
        if col not in x_out.columns:
            x_out[col] = 0

    unscaled = x_out[feature_columns].fillna(0)
    return scaler.transform(unscaled).astype(np.float32)
