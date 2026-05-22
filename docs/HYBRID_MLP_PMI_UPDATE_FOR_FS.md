# Hybrid MLP + PMI Rules Update untuk Tim Fullstack

Dokumen ini menjelaskan perubahan terbaru pada FastAPI AI service Emigria.

Perubahan utama:

```text
MLP tetap menjadi base model.
PMI rules menjadi domain adaptation layer untuk lowongan PMI Indonesia.
Output sekarang punya final risk score dalam persen.
Input contract dari Express/Gemini tetap sama.
```

---

## 1. Ringkasan Perubahan

File yang berubah:

```text
api/pmi_rules.py
api/app.py
data/pmi_seed_sources.csv
```

Perubahan pada `api/pmi_rules.py`:

```text
1. Menambahkan hybrid final risk score:
   final_score = 0.3 * ml_score + 0.7 * pmi_score

2. Menambahkan output persen:
   - ml_fraud_percentage
   - pmi_risk_percentage
   - final_risk_percentage

3. Menambahkan keyword red flag PMI:
   - biaya administrasi / biaya admin
   - paspor asli / paspor ditahan / paspor disimpan
   - dokumen asli
   - legalitas dalam proses
   - izin operasional dalam proses
   - tanpa prosedur resmi / tanpa tes resmi
   - casino online / judi online / scam center
   - mess tertutup / dilarang keluar
   - gaji puluhan juta / bonus target / fasilitas mewah
   - Kamboja, Myanmar, Laos, Macau

4. Menambahkan hard-stop logic:
   Jika dua atau lebih red flag kritis muncul bersamaan, hasil minimal masuk HIGH_RISK.

5. Menambahkan legit signal:
   Sinyal seperti BP2MI, P3MI, SIPMI, SIPP TKIS, izin operasional, kontrak resmi,
   dan tidak ada biaya penempatan dapat mengurangi PMI rule score.
```

Perubahan pada `api/app.py`:

```text
pmi_rule_threshold diset menjadi 6 saat memanggil combine_ml_and_pmi_rules().
```

File `data/pmi_seed_sources.csv`:

```text
Berisi daftar sumber awal untuk membangun dataset PMI real di masa depan.
File ini bukan input runtime FastAPI.
```

---

## 2. Input Contract Tetap Sama

Express tetap mengirim JSON hasil ekstraksi Gemini seperti sebelumnya.

Minimal input:

```json
{
  "title": "",
  "location": "",
  "country": "",
  "salary_range": "",
  "description": "",
  "requirements": "",
  "company_profile": "",
  "employment_type": "Unknown",
  "industry": "Unknown",
  "benefits": "",
  "required_experience": "Not Specified",
  "required_education": "Not Specified",
  "telecommuting": 0,
  "has_company_logo": 0,
  "has_questions": 0
}
```

Field tambahan seperti `salary_currency` boleh tetap ada di backend utama, tetapi FastAPI model hanya memakai field yang relevan.

PMI rules membaca teks dari:

```text
title
description
requirements
benefits
company_profile
salary_range
has_company_logo
has_questions
```

---

## 3. Risk Signals dari Gemini

`risk_signals` dari Gemini tidak wajib untuk FastAPI AI service.

Status saat ini:

```text
extracted_data = wajib
risk_signals   = opsional
```

Tetap disarankan disimpan di backend karena berguna untuk:

```text
1. explanation UI
2. debugging hasil OCR/Gemini
3. logging scan
4. pengembangan rules berikutnya
```

Jika ingin dikirim ke FastAPI, masukkan sebagai `extra`, tetapi versi rules saat ini belum bergantung langsung pada `extra.risk_signals`.

---

## 4. Formula Final Risk

FastAPI sekarang menghitung:

```text
ml_score  = output MLP, range 0.0 - 1.0
pmi_score = pmi_rule_score dinormalisasi ke 0.0 - 1.0

weighted_score = (0.3 * ml_score) + (0.7 * pmi_score)
```

Kemudian ada safety adjustment:

```text
Jika MLP melewati threshold model:
final_score = max(weighted_score, ml_score)

Jika hard-stop red flag aktif:
final_score = max(final_score, 0.60)
```

Artinya:

```text
Kalau MLP tinggi, final risk mengikuti MLP.
Kalau PMI rules tinggi, final risk mengikuti PMI.
Kalau red flag kritis muncul, sistem tidak membiarkan skor terlalu rendah.
```

---

## 5. Response Baru dari FastAPI

Contoh response:

```json
{
  "ml_fraud_probability": 0.6939886212348938,
  "ml_fraud_percentage": 69.4,
  "ml_fraud_prediction": 1,
  "pmi_rule_score": 2,
  "pmi_normalized_score": 0.2,
  "pmi_risk_percentage": 20.0,
  "pmi_rule_prediction": 0,
  "triggered_rules": [
    "no_company_logo",
    "no_screening_questions"
  ],
  "hard_stop_triggered": false,
  "hard_stop_count": 0,
  "ml_weight": 0.3,
  "pmi_weight": 0.7,
  "final_risk_score": 0.694,
  "final_risk_percentage": 69.4,
  "fraud_prediction": 1,
  "risk_level": "HIGH_RISK",
  "threshold": 0.5,
  "pmi_rule_threshold": 6,
  "review_threshold": 0.4,
  "high_risk_threshold": 0.6
}
```

---

## 6. Field yang Disarankan untuk FE

FE tidak perlu menampilkan semua field.

Field utama:

```json
{
  "risk_level": "HIGH_RISK",
  "final_risk_percentage": 69.4,
  "ml_fraud_percentage": 69.4,
  "pmi_risk_percentage": 20.0,
  "triggered_rules": [
    "no_company_logo",
    "no_screening_questions"
  ]
}
```

Rekomendasi tampilan:

```text
Status Risiko: Berisiko Tinggi
Skor Risiko: 69.4%

Rincian Analisis:
- Skor Model AI: 69.4%
- Skor Red Flag PMI: 20.0%

Alasan Terdeteksi:
- Tidak ada logo perusahaan
- Tidak ada pertanyaan/screening
```

Field debug/admin yang tidak perlu tampil ke user biasa:

```text
ml_weight
pmi_weight
threshold
pmi_rule_threshold
hard_stop_count
pmi_normalized_score
final_risk_score
ml_fraud_probability
```

---

## 7. Catatan Risk Level

Kode saat ini masih mendukung tiga level internal:

```text
LOW_RISK  : final_risk_score < 0.40
REVIEW    : 0.40 <= final_risk_score < 0.60
HIGH_RISK : final_risk_score >= 0.60
```

Namun untuk UI produk yang hanya ingin dua label, FE bisa mapping:

```text
LOW_RISK -> Risiko Rendah
REVIEW -> Berisiko / Perlu Dicek
HIGH_RISK -> Berisiko Tinggi
```

Jika tim ingin backend hanya mengirim `LOW_RISK` dan `HIGH_RISK`, AI service perlu diubah lagi menjadi threshold final 0.50:

```text
final_risk_percentage < 50  -> LOW_RISK
final_risk_percentage >= 50 -> HIGH_RISK
```

---

## 8. Contoh Hasil Test

### 8.1 Poster Oranye TKW

Input ringkas:

```text
Lowongan TKW, penempatan Hongkong/Taiwan/Malaysia/Singapore,
gaji 10 jt per bulan, PT BMC LPK Teguh Agensi Sukses.
```

Response:

```json
{
  "ml_fraud_percentage": 69.4,
  "pmi_risk_percentage": 20.0,
  "final_risk_percentage": 69.4,
  "risk_level": "HIGH_RISK",
  "triggered_rules": [
    "no_company_logo",
    "no_screening_questions"
  ]
}
```

Interpretasi:

```text
MLP mendeteksi risiko tinggi.
PMI rules hanya mendeteksi sinyal ringan.
Final risk mengikuti skor MLP karena MLP melewati threshold.
```

### 8.2 Loker Admin Casino Online Macau

Input ringkas:

```text
Admin Casino Online, gaji 35000 HKD/bulan,
biaya administrasi 15 juta di awal,
menyerahkan paspor asli, izin operasional dalam proses,
tanpa tes resmi.
```

Response:

```json
{
  "ml_fraud_percentage": 0.64,
  "pmi_risk_percentage": 100.0,
  "hard_stop_triggered": true,
  "hard_stop_count": 7,
  "final_risk_percentage": 70.19,
  "risk_level": "HIGH_RISK",
  "triggered_rules": [
    "biaya administrasi",
    "biaya admin",
    "proses keberangkatan cepat",
    "menyerahkan paspor",
    "paspor asli",
    "dokumen asli",
    "izin operasional dalam proses",
    "tanpa tes resmi",
    "casino online",
    "tidak perlu pengalaman",
    "tidak memerlukan pengalaman",
    "tidak perlu pendidikan",
    "no_company_logo"
  ]
}
```

Interpretasi:

```text
MLP rendah karena model baseline dilatih dari dataset English.
PMI rules menangkap red flag lokal yang berat.
Final risk menjadi HIGH_RISK karena PMI rules dan hard-stop.
```

### 8.3 Poster Merah PT Mulia Laksana Sejahtera

Input ringkas:

```text
Lowongan luar negeri khusus wanita,
tujuan Malaysia/Singapore/Taiwan/Hongkong,
proses dibiayai PT, kontak personal.
```

Response:

```json
{
  "ml_fraud_percentage": 0.03,
  "pmi_risk_percentage": 10.0,
  "final_risk_percentage": 7.01,
  "risk_level": "LOW_RISK",
  "triggered_rules": [
    "no_screening_questions"
  ]
}
```

Interpretasi:

```text
Tidak ada red flag berat seperti biaya admin, visa turis,
paspor ditahan, legalitas dalam proses, atau gaji fantastis.
```

### 8.4 Loker Hong Kong PLRT dengan SIPP TKIS

Input ringkas:

```text
Penata Laksana Rumah Tangga Hong Kong,
gaji 4600 HKD/bulan, company profile jelas,
ada SIPP TKIS dan izin operasional.
```

Response:

```json
{
  "ml_fraud_percentage": 5.33,
  "pmi_risk_percentage": 0.0,
  "final_risk_percentage": 1.6,
  "risk_level": "LOW_RISK",
  "triggered_rules": [
    "no_company_logo",
    "no_screening_questions",
    "legit_signal_count:2"
  ]
}
```

Interpretasi:

```text
Sinyal resmi seperti SIPP TKIS dan izin operasional menurunkan PMI score.
```

---

## 9. Deployment Notes

Jika AI service dijalankan via Docker Compose, tim FS perlu rebuild/restart:

```bash
docker compose up --build
```

Atau lebih bersih:

```bash
docker compose down
docker compose up --build
```

Jika jalan lokal tanpa Docker:

```bash
uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
```

Perubahan ini adalah perubahan source code Python, bukan artifact model.

---

## 10. Summary untuk FE

Yang perlu FE pakai:

```text
final_risk_percentage -> skor utama di UI
risk_level            -> label status
ml_fraud_percentage   -> skor model AI
pmi_risk_percentage   -> skor red flag PMI
triggered_rules       -> alasan/penjelasan
```

Yang perlu diingat:

```text
Input contract tidak berubah.
Output response bertambah field baru.
Risk signals dari Gemini tetap opsional.
Geo risk, reality check, dan smart action tetap boleh dikelola di backend utama.
```
