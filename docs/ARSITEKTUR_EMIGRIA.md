# Arsitektur Teknis Emigria — AI Pipeline

## Ringkasan Proyek

**Emigria** adalah platform deteksi penipuan lowongan kerja luar negeri untuk calon Pekerja Migran Indonesia (PMI). Menggunakan arsitektur **Two-Stage AI Pipeline**:
- **Stage 1**: Gemini API 1.5 Flash → ekstraksi data dari brosur (gambar/teks)
- **Stage 2**: Model MLP (Deep Learning) → klasifikasi fraud/legitimate

---

## Alur Sistem End-to-End

```
User upload brosur (gambar/teks/URL)
        ↓
[Express.js - Orchestrator]
        ↓
Kirim gambar ke Gemini API 1.5 Flash
        ↓
Gemini ekstrak → JSON terstruktur (14 field)
        ↓
Express.js kirim JSON ke FastAPI
        ↓
[FastAPI - AI Service]
  ├── create_features() → feature engineering
  ├── StandardScaler → scaling
  └── MLP Model (.keras) → fraud_score (0.0 - 1.0)
        ↓
Return ke Express.js
        ↓
Express.js gabungkan:
  ├── Fraud Score (dari FastAPI)
  ├── Geo Risk (lookup statis)
  └── Reality Check (lookup statis)
        ↓
Kirim ke Frontend → User lihat hasil
```

---

## Pembagian Tanggung Jawab

| Komponen | Penanggung Jawab | Teknologi |
|---|---|---|
| Dataset & Preprocessing | Data Science (DS) | Pandas, Sklearn |
| Model MLP + FastAPI | AI Engineer (AI) | TensorFlow, FastAPI |
| Backend Orchestrator | Full-Stack (FS) | Express.js |
| Frontend UI | Full-Stack (FS) | Vite/React |
| Gemini API Integration | Full-Stack (FS) | Gemini 1.5 Flash |
| Data Statis (Geo Risk, Reality Check) | DS + FS | JSON lookup |

---

## Struktur JSON dari Gemini API

Gemini diperintahkan (via prompt) untuk mengekstrak data brosur dalam format berikut. Format ini **kita yang tentukan**, bukan Gemini yang bebas output.

```json
{
  "title": "Cleaning Service",
  "location": "Malaysia, Kuala Lumpur",
  "country": "Malaysia",
  "employment_type": "Full-time",
  "required_experience": "Not Applicable",
  "required_education": "High School or equivalent",
  "industry": "Consumer Services",
  "salary_range": "1500-2000",
  "company_profile": "",
  "description": "Gaji 20jt, langsung berangkat, visa ziarah...",
  "requirements": "",
  "benefits": "Makan gratis, tempat tinggal",
  "telecommuting": 0,
  "has_company_logo": 0,
  "has_questions": 0
}
```

### Kenapa Format Ini?
- Setiap field **mapping langsung** ke kolom raw di dataset training (`fake_job_postings.csv`)
- `create_features()` bisa langsung proses tanpa adaptasi
- Jika field tidak ditemukan di brosur → isi string kosong `""` atau `0`
- **Field kosong = sinyal juga** (loker fraud biasanya minim info)

### Field Wajib vs Opsional

| Field | Wajib? | Alasan |
|---|---|---|
| `title` | ✅ | Untuk `title_length` |
| `description` | ✅ | Fitur terpenting: `desc_length` + `scam_keyword_score` |
| `company_profile` | ✅ | Fitur #1 paling berpengaruh di model |
| `has_company_logo` | ✅ | Fitur top di XGBoost |
| `country` | ✅ | Untuk `country_fraud_rate` |
| `employment_type` | ✅ | Untuk encoding |
| `industry` | ✅ | Untuk `industry_fraud_rate` |
| `salary_range` | ⚠️ Opsional | Default `""` jika tidak ada |
| `requirements` | ⚠️ Opsional | Default `""` |
| `benefits` | ⚠️ Opsional | Default `""` |
| `telecommuting` | ⚠️ Opsional | Default `0` |
| `has_questions` | ⚠️ Opsional | Default `0` |

---

## Model MLP (AI Engineer Scope)

### Spesifikasi

| Aspek | Detail |
|---|---|
| Arsitektur | Multi-Layer Perceptron (MLP) |
| Framework | TensorFlow - Functional API |
| Loss Function | **Custom Focal Loss** (handle class imbalance 95:5) |
| Custom Callback | F1 Score Monitor per epoch |
| Input | ~20 fitur numerik (hasil `create_features()` + scaling) |
| Output | 1 neuron, sigmoid → skor 0.0 - 1.0 |
| Export | `.keras` format |
| Benchmark | Harus ≥ F1 74% (XGB Tuned dari tim DS) |

### Kriteria Main Quest AI Engineer

- [x] Model Deep Learning menggunakan TensorFlow Functional API
- [x] Minimal 1 komponen kustom (Custom Focal Loss)
- [x] Export model `.keras` siap produksi
- [x] Kode inference sederhana

### Dari Mana Data & Fitur?

**Dari notebook tim DS** — kita LANJUTKAN, bukan bikin ulang:
- Dataset: `fake_job_postings.csv` (17.880 baris)
- Preprocessing: `parse_salary()`, `COUNTRY_MAP`, cleaning
- Feature Engineering: `create_features()` → 20+ fitur numerik
- Split: 80/20 stratified, random_state=42
- Scaling: StandardScaler

Kita hanya **ganti bagian modeling** (RF/XGB → MLP).

---

## FastAPI — AI Inference Service

### Endpoint

```
POST /predict
Content-Type: application/json
```

### Request Body
```json
{
  "title": "Cleaning Service",
  "country": "Malaysia",
  "description": "Gaji 20jt...",
  "company_profile": "",
  "has_company_logo": 0,
  ...
}
```

### Response
```json
{
  "fraud_score": 0.87,
  "fraud_label": "HIGH RISK"
}
```

### Threshold Label

| Score | Label |
|---|---|
| ≥ 0.7 | HIGH RISK |
| 0.4 - 0.7 | MEDIUM RISK |
| < 0.4 | LOW RISK |

### File yang Di-deploy

```
📁 api/
├── app.py                 ← FastAPI main
├── preprocessing.py       ← create_features() untuk production
├── requirements.txt       ← tensorflow, fastapi, joblib, uvicorn
├── Dockerfile             ← Untuk deploy
└── model/
    ├── emigria_mlp.keras  ← Model MLP
    ├── scaler.joblib      ← StandardScaler
    └── encoder.joblib     ← TargetEncoder
```

---

## Geo Risk — Data Statis (Lookup)

**Lokasi**: Express.js atau Frontend (bukan FastAPI)
**Sifat**: JSON statis, lookup by country

### Contoh Data

```json
{
  "Malaysia": {
    "fraud_rate_pct": 57.1,
    "crime_index": 62,
    "nearest_kbri": "Kuala Lumpur",
    "risk_level": "HIGH"
  },
  "Singapore": {
    "fraud_rate_pct": 0,
    "crime_index": 16,
    "nearest_kbri": "Singapore City",
    "risk_level": "LOW"
  },
  "Taiwan": {
    "fraud_rate_pct": 0,
    "crime_index": 15,
    "nearest_kbri": "Taipei",
    "risk_level": "LOW"
  }
}
```

### Data yang Perlu Dikumpulkan (oleh DS/FS)
- Lokasi & jumlah KBRI per negara
- Crime index per negara (sumber: Numbeo / UNODC)
- Fraud rate dari dataset (sudah ada di notebook DS)

---

## Reality Check — Data Statis (Lookup)

**Lokasi**: Express.js atau Frontend (bukan FastAPI)
**Sifat**: JSON statis, lookup by country

### Contoh Data

```json
{
  "Malaysia": {
    "currency": "MYR",
    "salary_min_idr": 5000000,
    "salary_max_idr": 8000000,
    "source": "BP2MI 2024"
  },
  "Hong Kong": {
    "currency": "HKD",
    "salary_min_idr": 7000000,
    "salary_max_idr": 10000000,
    "source": "Minimum Allowable Wage HK 2024"
  },
  "Saudi Arabia": {
    "currency": "SAR",
    "salary_min_idr": 4000000,
    "salary_max_idr": 7000000,
    "source": "BP2MI 2024"
  }
}
```

### Logic Perbandingan (di Express.js)

```javascript
const claimed = extractedSalaryFromGemini; // dari brosur
const standard = realityCheckData[country];

if (claimed > standard.salary_max_idr * 2) {
  verdict = "UNREALISTIC";
} else if (claimed > standard.salary_max_idr) {
  verdict = "SUSPICIOUS";
} else {
  verdict = "REASONABLE";
}
```

### Sumber Data
- Website BP2MI (standar gaji per negara tujuan)
- Regulasi upah minimum negara tujuan
- Data hanya untuk 10-15 negara tujuan utama PMI

---

## Tampilan Hasil di UI (Contoh)

```
🔴 PERINGATAN PENIPUAN — Skor: 87% (HIGH RISK)

📍 Geo Risk:
   - Negara: Malaysia
   - Crime Index: 62/100
   - KBRI terdekat: Kuala Lumpur
   - Risk Level: HIGH

💰 Reality Check:
   - Klaim gaji (dari brosur): Rp 25.000.000/bulan
   - Standar BP2MI Malaysia: Rp 5.000.000 - 8.000.000/bulan
   - ❌ Tidak realistis (3x lipat standar)

🤖 Smart Action:
   - [Laporkan ke BP2MI]
   - [Bagikan ke WhatsApp]
```

---

## Ringkasan Arsitektur

| Komponen | Teknologi | Lokasi | Sifat |
|---|---|---|---|
| Ekstraksi brosur | Gemini 1.5 Flash | Dipanggil dari Express.js | API call |
| Fraud Detection | MLP TensorFlow | FastAPI | Model DL |
| Geo Risk | JSON lookup | Express.js / Frontend | Data statis |
| Reality Check | JSON lookup | Express.js / Frontend | Data statis |
| Orchestrator | Express.js | Backend utama | Routing & logic |
| UI | Vite/React | Frontend | Tampilan user |
| Database | Neon (PostgreSQL) | Cloud | Log anonim |

---

## Next Step

1. **AI Engineer**: Mulai training MLP → export `.keras` → bikin FastAPI
2. **DS**: Siapkan data statis Geo Risk & Reality Check (KBRI, crime index, standar gaji BP2MI)
3. **FS**: Setup Express.js orchestrator + integrasi Gemini API + frontend UI
