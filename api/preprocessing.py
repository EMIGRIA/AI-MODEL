"""Preprocessing utilities mirroring the Emigria DS training pipeline (V2 PMI-adapted)."""

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

# V2: 5 fitur PMI baru
PMI_FEATURES = [
    "has_visa_wisata",
    "has_direct_contact",
    "is_pmi_risk_destination",
    "has_p3mi_signal",
    "pmi_risk_score",
]

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
] + PMI_FEATURES

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
    "KH": "Cambodia",
    "LA": "Laos",
    "MO": "Macau",
}

# V2 expanded keyword lists
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
    "visa on arrival",
    "berangkat hari ini",
    "berangkat minggu ini",
    "tanpa kontrak",
    "proses kilat",
    "rekrut langsung",
    "tidak ada potongan",
    "gaji langsung diterima",
    "dp dulu",
    "uang muka",
    "biaya keberangkatan",
    "paspor dipegang agen",
    "paspor ditahan",
    "overstay",
    "illegal working",
    "working permit tidak diurus",
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
    "gaji dollar",
    "gaji ringgit",
    "gaji dolar",
    "kerja malaysia",
    "kerja singapura",
    "kerja taiwan",
    "tkw",
    "tki",
    "cpmi",
    "calon pmi",
    "sponsor visa",
    "agency fee",
    "recruitment fee",
    "contract 2 tahun",
    "contract 3 tahun",
    "penampungan",
    "training center",
    "slot terbatas",
    "quota terbatas",
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
    "wa kami",
    "hubungi wa",
    "chat wa",
    "dm kami",
    "inbox kami",
    "dm untuk info",
    "klik link",
    "info lebih lanjut hubungi",
    "untuk informasi hubungi",
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
    "perusahaan",
    "kandidat",
    "pengalaman",
    "ijazah",
    "kualifikasi",
    "posisi",
    "jabatan",
    "lamaran",
    "berkas",
    "cv",
    "dengan",
    "untuk",
    "dari",
    "yang",
    "dan",
    "tidak",
    "ada",
    "anda",
    "akan",
]

# V2: Negara TPPO hotspot
PMI_RISK_COUNTRIES = {
    "cambodia": 0.80,
    "kamboja": 0.80,
    "myanmar": 0.75,
    "laos": 0.70,
    "lao": 0.70,
    "vietnam": 0.50,
    "thailand": 0.45,
    "china": 0.40,
    "tiongkok": 0.40,
    "macau": 0.75,
    "macao": 0.75,
    "malaysia": 0.30,
    "taiwan": 0.30,
    "hong kong": 0.25,
    "saudi arabia": 0.25,
    "arab saudi": 0.25,
    "uae": 0.20,
    "dubai": 0.20,
    "qatar": 0.20,
    "bahrain": 0.20,
    "kuwait": 0.20,
    "oman": 0.20,
    "turkey": 0.35,
    "turki": 0.35,
    "greece": 0.30,
    "yunani": 0.30,
    "philippines": 0.20,
    "filipina": 0.20,
}

# V2: Sinyal loker PMI resmi
PMI_LEGIT_SIGNALS = [
    "bp2mi",
    "bnp2tki",
    "p3mi",
    "sipmi",
    "sisnaker",
    "visa kerja",
    "work permit",
    "working visa",
    "kontrak kerja resmi",
    "perjanjian penempatan",
    "asuransi pmi",
    "bpjs ketenagakerjaan",
    "terdaftar di kemnaker",
    "berizin kemnaker",
    "id.bp2mi.go.id",
    "kemnaker.go.id",
    "tidak ada biaya penempatan",
    "biaya ditanggung majikan",
    "sipp tkis",
    "izin operasional",
]

# V2: Country prior fraud rate
COUNTRY_PRIOR_FRAUD_RATE = {
    "Cambodia": 0.75,
    "Myanmar": 0.70,
    "Laos": 0.65,
    "Macau": 0.75,
    "Macao": 0.75,
    "China": 0.40,
    "Tiongkok": 0.40,
    "Vietnam": 0.45,
    "Thailand": 0.40,
    "Turkey": 0.35,
    "Greece": 0.30,
    "Malaysia": 0.28,
    "Taiwan": 0.25,
    "Hong Kong": 0.22,
    "Saudi Arabia": 0.22,
    "UAE": 0.18,
    "Qatar": 0.18,
    "Kuwait": 0.18,
    "Bahrain": 0.18,
    "Oman": 0.16,
    "Indonesia": 0.15,
    "Philippines": 0.18,
    "United States": 0.048,
    "United Kingdom": 0.020,
    "Germany": 0.015,
    "Australia": 0.035,
    "New Zealand": 0.015,
    "Canada": 0.020,
    "Singapore": 0.030,
}

PRIOR_BLEND_ALPHA = 0.6


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


def calc_pmi_risk_score(text: Any) -> int:
    """V2: Composite PMI risk score (0-10) — sinyal khusus konteks PMI Indonesia."""
    if not isinstance(text, str):
        return 0
    lowered = text.lower()
    score = 0

    visa_traps = [
        "visa wisata",
        "visa turis",
        "tourist visa",
        "visa on arrival",
        "bebas visa",
        "voa",
    ]
    score += 4 * sum(1 for v in visa_traps if v in lowered)

    fee_traps = [
        "biaya administrasi",
        "biaya keberangkatan",
        "bayar dulu",
        "dp dulu",
        "uang muka",
        "advance payment",
        "upfront",
    ]
    score += 3 * sum(1 for f in fee_traps if f in lowered)

    tier1 = ["kamboja", "cambodia", "myanmar", "laos", "burma"]
    score += 3 * sum(1 for c in tier1 if c in lowered)

    muluk = [
        "gaji besar",
        "penghasilan jutaan",
        "income tinggi",
        "langsung diterima",
        "proses cepat",
        "tanpa syarat",
    ]
    score += 1 * sum(1 for m in muluk if m in lowered)

    informal_contact = [
        "wa ",
        "whatsapp",
        "telegram",
        "t.me",
        "wa.me",
        "hubungi hp",
        "sms ke",
    ]
    score += 2 * sum(1 for c in informal_contact if c in lowered)

    return min(score, 10)


def extract_pmi_features_from_row(row: pd.Series) -> dict[str, int]:
    """V2: Ekstrak 5 fitur PMI domain-specific dari satu baris dataframe."""
    desc = str(row.get("description", "")).lower()
    title = str(row.get("title", "")).lower()
    req = str(row.get("requirements", "")).lower()
    loc = str(row.get("location", "")).lower()
    full_text = f"{title} {desc} {req} {loc}"

    visa_patterns = [
        "visa wisata",
        "visa turis",
        "tourist visa",
        "visa on arrival",
        "bebas visa asean",
        "without work permit",
        "tanpa work permit",
    ]
    has_visa_wisata = int(any(p in full_text for p in visa_patterns))

    contact_patterns = [
        "wa.me",
        "t.me",
        "telegram",
        "whatsapp",
        "wa kami",
        "chat wa",
        "hubungi wa",
    ]
    has_direct_contact = int(
        any(p in full_text for p in contact_patterns)
        or bool(re.search(r"\+62\s*8[0-9]{8,11}", full_text))
    )

    is_pmi_risk_destination = int(
        any(country in full_text for country in PMI_RISK_COUNTRIES.keys())
    )

    has_p3mi_signal = int(
        any(signal in full_text for signal in PMI_LEGIT_SIGNALS)
    )

    pmi_risk_score = calc_pmi_risk_score(full_text)

    return {
        "has_visa_wisata": has_visa_wisata,
        "has_direct_contact": has_direct_contact,
        "is_pmi_risk_destination": is_pmi_risk_destination,
        "has_p3mi_signal": has_p3mi_signal,
        "pmi_risk_score": pmi_risk_score,
    }


def prepare_base_dataframe(raw_df: pd.DataFrame, has_target: bool = False) -> pd.DataFrame:
    """V2: Cleaning + feature engineering + 5 fitur PMI."""
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

    # V2: cek title+desc+req, threshold 2 (turun dari 3)
    df["is_indonesian_posting"] = df.apply(
        lambda row: int(
            sum(
                word
                in (
                    str(row["title"])
                    + " "
                    + str(row["description"])
                    + " "
                    + str(row["requirements"])
                ).lower()
                for word in ID_WORDS
            )
            >= 2
        ),
        axis=1,
    )

    # V2: tambah 5 fitur PMI
    pmi_feats = df.apply(extract_pmi_features_from_row, axis=1)
    pmi_df = pd.DataFrame(pmi_feats.tolist(), index=df.index)
    df = pd.concat([df, pmi_df], axis=1)

    return df


def transform_features(
    base_df: pd.DataFrame,
    target_encoder: Any,
    tfidf_vectorizer: Any,
    scaler: Any,
    country_salary_data: dict[str, Any],
    feature_columns: list[str] | None = None,
) -> np.ndarray:
    """V2: Transform features dengan country prior blending + combined-text TF-IDF."""
    feature_columns = feature_columns or FEATURE_COLUMNS
    x_out = base_df.copy()

    encoded = target_encoder.transform(x_out[TARGET_ENCODE_COLS])
    x_out["country_fraud_rate"] = encoded["country"]
    x_out["emp_type_fraud_rate"] = encoded["employment_type"]
    x_out["industry_fraud_rate"] = encoded["industry"]

    # V2: Country prior blending (case-insensitive lookup untuk robustness)
    _prior_lookup = {k.lower(): v for k, v in COUNTRY_PRIOR_FRAUD_RATE.items()}

    def blend_country_rate(row: pd.Series) -> float:
        country = str(row["country"]).strip().lower()
        te_rate = row["country_fraud_rate"]
        if country in _prior_lookup:
            prior = _prior_lookup[country]
            return PRIOR_BLEND_ALPHA * prior + (1 - PRIOR_BLEND_ALPHA) * te_rate
        return te_rate

    x_out["country_fraud_rate"] = x_out.apply(blend_country_rate, axis=1)
    x_out["country_safety_score"] = 1 - x_out["country_fraud_rate"]

    country_salary_avg = country_salary_data["country_salary_avg"]
    global_salary_median = country_salary_data["global_salary_median"]

    x_out["country_avg_salary"] = (
        x_out["country"].map(country_salary_avg).fillna(global_salary_median)
    )
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

    # V2 FIX: TF-IDF dari combined text (title + desc + req)
    combined_text = (
        x_out["title"].fillna("")
        + " "
        + x_out["description"].fillna("")
        + " "
        + x_out["requirements"].fillna("")
    )
    tfidf_matrix = tfidf_vectorizer.transform(combined_text).toarray()
    tfidf_df = pd.DataFrame(tfidf_matrix, columns=TFIDF_FEATURES, index=x_out.index)
    x_out = pd.concat([x_out, tfidf_df], axis=1)

    for col in feature_columns:
        if col not in x_out.columns:
            x_out[col] = 0

    unscaled = x_out[feature_columns].fillna(0)
    return scaler.transform(unscaled).astype(np.float32)
