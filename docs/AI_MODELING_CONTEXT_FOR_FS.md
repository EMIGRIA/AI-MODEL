# Konteks AI Modeling Emigria untuk Tim Fullstack

Dokumen ini merangkum keputusan, masalah, dan solusi dari proses modeling AI Emigria. Tujuannya supaya tim Fullstack memahami kenapa FastAPI AI service dibuat seperti sekarang, kenapa ada model MLP, kenapa ada PMI rule layer, dan bagaimana Express.js sebaiknya mengirim data hasil Gemini.

---

## 1. Tujuan Sistem AI Emigria

Emigria adalah aplikasi web untuk mendeteksi potensi penipuan pada lowongan kerja luar negeri, terutama untuk calon Pekerja Migran Indonesia.

Arsitektur besar:

```text
Frontend React/Vite
-> Express.js orchestrator
-> Gemini API untuk ekstraksi brosur/gambar/teks menjadi JSON
-> FastAPI AI service
-> MLP TensorFlow + PMI risk rules
-> hasil risiko dikembalikan ke Express/frontend
```

AI service bertugas menerima JSON lowongan kerja dari Express, lalu mengembalikan:

```text
fraud probability dari MLP
PMI rule score
final fraud prediction
risk level
alasan/red flags
```

---

## 2. Dataset dan Masalah Awal

Awalnya ada dua dataset/file penting:

```text
fake_job_postings.csv
main_data.csv
```

`main_data.csv` terlihat seperti dataset siap training, tetapi setelah dianalisis ternyata file itu adalah output akhir notebook Data Science untuk dashboard.

Masalah `main_data.csv`:

```text
1. Banyak fitur sudah di-scale.
2. Ada kolom output model lama:
   - fraud_prediction
   - fraud_probability
   - risk_level
3. Ada potensi data leakage jika kolom output tersebut dipakai training.
4. Kurang ideal untuk FastAPI karena input real dari Gemini masih berbentuk data mentah.
```

Karena itu pipeline final AI Engineer tidak memakai `main_data.csv` sebagai sumber utama training.

Keputusan final:

```text
Training MLP dimulai dari fake_job_postings.csv
dan mirror preprocessing notebook final tim Data Science.
```

---

## 3. Kenapa Mirror Notebook Data Science

Notebook DS membangun fitur dari raw job posting dengan tahapan:

```text
fake_job_postings.csv
-> cleaning
-> salary parsing
-> country extraction
-> feature engineering
-> target encoding
-> TF-IDF description
-> StandardScaler
-> RF/XGBoost
```

Di sisi AI Engineer, classifier RF/XGBoost diganti menjadi MLP TensorFlow, tetapi preprocessing tetap dimirror agar konsisten.

Pipeline final:

```text
fake_job_postings.csv
-> cleaning sesuai DS
-> feature engineering sesuai DS
-> train/validation/test split stratified
-> target encoding leakage-free
-> TF-IDF description 100 fitur
-> StandardScaler
-> MLP TensorFlow
-> threshold tuning
-> export model dan artifacts
```

Total fitur MLP:

```text
22 fitur engineered + 100 TF-IDF = 122 fitur
```

---

## 4. Model MLP yang Dibangun

Model menggunakan TensorFlow Functional API.

Arsitektur:

```text
Input(122)
-> Dense(128, relu)
-> BatchNormalization
-> Dropout(0.30)
-> Dense(64, relu)
-> BatchNormalization
-> Dropout(0.20)
-> Dense(32, relu)
-> Dropout(0.10)
-> Dense(1, sigmoid)
```

Karena dataset sangat imbalanced, training memakai:

```text
class_weight
```

Model juga memakai custom callback:

```text
FraudMetricsCallback
```

Callback ini menghitung precision, recall, dan F1 fraud pada validation set setiap epoch.

---

## 5. Hasil Evaluasi Final

Threshold produksi MVP dipilih:

```text
0.50
```

Hasil final di test set:

```text
Accuracy        : 0.97
Precision fraud : 0.69
Recall fraud    : 0.73
F1 fraud        : 0.71
ROC-AUC         : 0.9198
PR-AUC          : 0.7625
MAE             : 0.0303
```

Confusion matrix:

```text
TN = 3258
FP = 53
FN = 44
TP = 119
```

Interpretasi:

```text
Dari 163 lowongan fraud di test set, model berhasil menangkap 119.
Ada 44 fraud yang masih lolos.
False alarm pada non-fraud sebanyak 53 dari 3311.
```

Model ini cukup baik sebagai baseline MLP untuk fraud detection umum.

---

## 6. Masalah Penting: Domain Mismatch

Setelah model dicoba dengan contoh lowongan PMI Indonesia seperti:

```text
Gaji besar, proses cepat, visa turis, segera hubungi sekarang,
tanpa pengalaman, company profile kosong.
```

Model MLP tetap memberi:

```text
LOW_RISK
```

Bahkan setelah teks diterjemahkan ke bahasa Inggris, model masih memberi probability sangat rendah.

Kesimpulan:

```text
Masalahnya bukan hanya bahasa.
Masalah utamanya adalah domain mismatch.
```

Dataset training:

```text
fake_job_postings.csv
```

lebih banyak berisi pola job posting internasional/English umum.

Use case Emigria:

```text
lowongan PMI Indonesia, brosur, WhatsApp, Facebook, TikTok,
visa turis, biaya admin, kirim paspor, langsung berangkat.
```

Pola PMI lokal belum cukup terwakili di dataset training.

---

## 7. Solusi: Hybrid MLP + PMI Rule Layer

Karena MLP tidak cukup kuat untuk red flag PMI Indonesia, AI service dibuat hybrid:

```text
MLP probability
+
PMI rule-based risk score
=
final fraud prediction
```

MLP berfungsi sebagai:

```text
fraud detector umum berbasis pola historis job posting
```

PMI rule layer berfungsi sebagai:

```text
safety layer untuk red flag lokal PMI Indonesia
```

Contoh red flag PMI:

```text
visa turis
visa ziarah
kirim paspor
bayar di muka
biaya administrasi
proses cepat
langsung berangkat
gaji besar
tanpa pengalaman
tanpa ijazah
kontak pribadi / Gmail / WhatsApp informal
company profile kosong
tidak ada logo perusahaan
tidak ada screening questions
```

Final logic:

```text
Jika MLP probability >= 0.50
atau PMI rule score >= 4
maka HIGH_RISK.

Selain itu LOW_RISK.
```

---

## 8. Kenapa Risk Level Hanya LOW dan HIGH

Awalnya ada rencana:

```text
LOW_RISK
MEDIUM_RISK
HIGH_RISK
```

Tapi untuk user akhir, label medium bisa membingungkan.

Keputusan final:

```text
LOW_RISK  -> Aman
HIGH_RISK -> Berisiko
```

Tidak ada `MEDIUM_RISK` di response final user-facing.

Jika probability sedikit di atas threshold, response tetap `HIGH_RISK`, tetapi frontend bisa menampilkan kata yang lebih ramah:

```text
Berisiko
```

bukan harus menampilkan:

```text
High Risk
```

---

## 9. Peran Gemini API

Gemini tidak langsung menentukan final fraud.

Peran Gemini:

```text
Mengekstrak informasi dari gambar/brosur/OCR/text menjadi JSON terstruktur.
```

Express.js mengirim hasil ekstraksi Gemini ke FastAPI.

Field yang dikirim harus konsisten.

Minimal JSON:

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

Jika Gemini tidak menemukan informasi, gunakan default di atas.

---

## 10. Risk Signals dari Gemini

Gemini juga disarankan mengekstrak risk signals, misalnya:

```json
{
  "extra": {
    "risk_signals": {
      "mentions_tourist_visa": true,
      "asks_for_passport": true,
      "asks_upfront_payment": false,
      "mentions_admin_fee": true,
      "promises_fast_process": true,
      "promises_high_salary": true,
      "no_experience_required": true,
      "company_identity_clear": false,
      "uses_personal_contact": true,
      "urgency_level": "high",
      "risk_keywords": [
        "visa turis",
        "biaya administrasi",
        "proses cepat"
      ]
    }
  }
}
```

Untuk versi FastAPI sekarang, PMI rule layer sudah bisa membaca raw text langsung. Risk signals dari Gemini bisa dipakai FS untuk UI/explanation atau pengembangan rule lanjutan.

---

## 11. Artifact Model

Artifact hasil training Colab ditaruh di:

```text
api/models/
```

File penting:

```text
emigria_mlp_model.keras
target_encoder.pkl
tfidf_vectorizer.pkl
standard_scaler.pkl
country_salary_avg.pkl
feature_columns.json
threshold.txt
preprocessing_config.json
metrics.json
threshold_tuning.csv
```

FastAPI akan load artifact tersebut saat startup.

---

## 12. Endpoint FastAPI

Endpoint:

```text
GET /health
POST /predict
POST /predict-batch
```

`/predict` untuk satu lowongan.

`/predict-batch` untuk banyak lowongan sekaligus.

Untuk MVP frontend, cukup pakai:

```text
POST /predict
```

---

## 13. Response FastAPI

Contoh response:

```json
{
  "ml_fraud_probability": 0.00001,
  "ml_fraud_prediction": 0,
  "pmi_rule_score": 13,
  "triggered_rules": [
    "visa turis",
    "proses cepat",
    "gaji besar",
    "tanpa pengalaman",
    "company_profile_empty",
    "no_company_logo"
  ],
  "fraud_prediction": 1,
  "risk_level": "HIGH_RISK",
  "threshold": 0.5,
  "pmi_rule_threshold": 4
}
```

Frontend sebaiknya fokus pada:

```text
fraud_prediction
risk_level
triggered_rules
```

MLP probability boleh disimpan untuk debugging/admin, tapi tidak wajib ditampilkan mentah ke user.

---

## 14. Kenapa Model Tetap Penting

Walaupun ada PMI rule layer, model MLP tetap penting.

Fungsi model:

```text
1. Memberikan probabilitas fraud berbasis data historis.
2. Menangkap pola umum yang tidak selalu bisa dibuat dengan keyword manual.
3. Menjadi core ML engine yang bisa dievaluasi secara objektif.
4. Bisa ditingkatkan jika nanti tersedia dataset PMI Indonesia.
```

PMI rule layer bukan pengganti model, tetapi pelengkap untuk konteks lokal.

---

## 15. Keterbatasan Saat Ini

Keterbatasan utama:

```text
Dataset belum spesifik PMI Indonesia.
```

Dampaknya:

```text
MLP kurang sensitif terhadap lowongan berbahasa Indonesia dan red flag lokal.
```

Solusi sementara:

```text
Hybrid MLP + PMI rules.
```

Solusi jangka panjang:

```text
Kumpulkan dataset lowongan PMI Indonesia asli,
label fraud/non-fraud,
lalu retrain model dengan data domain tersebut.
```

---

## 16. Ringkasan Untuk Tim FS

Yang perlu dipahami FS:

```text
1. Express/Gemini bertugas membuat JSON input yang konsisten.
2. FastAPI menerima JSON dan melakukan preprocessing.
3. MLP memberi fraud probability.
4. PMI rule layer menangkap red flag lokal.
5. Final output hanya LOW_RISK atau HIGH_RISK.
6. triggered_rules penting untuk menjelaskan alasan risiko ke user.
```

Dengan desain ini, Emigria tetap punya model ML yang valid, tetapi juga aman untuk konteks PMI Indonesia yang belum sepenuhnya tercakup di dataset training.

---

## 17. Key Contract Express ke FastAPI

Endpoint:

```text
POST /predict
Content-Type: application/json
```

Express harus mengirim JSON dengan key berikut:

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
  "has_questions": 0,
  "extra": {
    "risk_signals": {
      "mentions_tourist_visa": false,
      "mentions_pilgrimage_visa": false,
      "asks_for_passport": false,
      "asks_upfront_payment": false,
      "mentions_admin_fee": false,
      "promises_fast_process": false,
      "promises_high_salary": false,
      "no_experience_required": false,
      "no_degree_required": false,
      "uses_personal_contact": false,
      "company_identity_clear": false,
      "salary_claim_unrealistic": false,
      "urgency_level": "low",
      "risk_keywords": []
    }
  }
}
```

### 17.1 Field Wajib

Field berikut harus selalu dikirim, walaupun kosong:

```text
title
location
country
salary_range
description
requirements
company_profile
employment_type
industry
benefits
required_experience
required_education
telecommuting
has_company_logo
has_questions
```

Default value:

```text
Text kosong          -> ""
employment_type      -> "Unknown"
industry             -> "Unknown"
required_experience  -> "Not Specified"
required_education   -> "Not Specified"
telecommuting        -> 0
has_company_logo     -> 0
has_questions        -> 0
```

Binary value:

```text
0 = tidak / tidak ditemukan
1 = ya / ditemukan
```

### 17.2 Field Tambahan `extra.risk_signals`

`extra.risk_signals` direkomendasikan untuk dikirim dari Gemini/Express.

Untuk versi FastAPI sekarang:

```text
PMI rule layer sudah bisa membaca raw text langsung.
Risk signals tetap berguna untuk explanation, logging, dan pengembangan rule lanjutan.
```

Jangan ubah nama key tanpa koordinasi karena backend mengandalkan contract yang stabil.

---

## 18. Prompt Gemini untuk Ekstraksi JSON

Prompt ini bisa dipakai FS saat memanggil Gemini untuk mengekstrak brosur/gambar/OCR/text menjadi JSON.

Gunakan prompt ketat berikut:

```text
You are an information extraction system for Emigria, an app that detects possible fraud in overseas job offers for Indonesian migrant workers.

Your task is to extract structured JSON from the provided job poster, image, OCR text, or job description.

Return ONLY valid JSON. Do not include markdown, explanation, comments, or extra text.

Use this exact JSON schema:

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
  "has_questions": 0,
  "extra": {
    "risk_signals": {
      "mentions_tourist_visa": false,
      "mentions_pilgrimage_visa": false,
      "asks_for_passport": false,
      "asks_upfront_payment": false,
      "mentions_admin_fee": false,
      "promises_fast_process": false,
      "promises_high_salary": false,
      "no_experience_required": false,
      "no_degree_required": false,
      "uses_personal_contact": false,
      "company_identity_clear": false,
      "salary_claim_unrealistic": false,
      "urgency_level": "low",
      "risk_keywords": []
    }
  }
}

Rules:
1. Return valid JSON only.
2. Do not invent information that is not present.
3. If a field is missing, use the default value from the schema.
4. Keep text fields in the original language found in the poster.
5. For "country", infer the destination country if clearly mentioned.
6. For "salary_range", preserve the salary text exactly as written, including currency if available.
7. Set "has_company_logo" to 1 only if a company logo is visible or explicitly mentioned. Otherwise 0.
8. Set "has_questions" to 1 only if screening questions, interview questions, or application questions are present. Otherwise 0.
9. Set "telecommuting" to 1 only if the job is remote/work from home. Otherwise 0.
10. Set "company_identity_clear" to true only if company name, profile, address, or official identity is clear.
11. Set "uses_personal_contact" to true if the job uses WhatsApp, personal phone number, personal email, Gmail/Yahoo/Hotmail, or informal contact as the main application method.
12. Set "urgency_level" to one of: "low", "medium", "high".
13. Fill "risk_keywords" with exact suspicious phrases found in the input.

Risk signal definitions:
- "mentions_tourist_visa": true if text mentions visa turis, tourist visa, visa kunjungan, or similar.
- "mentions_pilgrimage_visa": true if text mentions visa ziarah, visa umroh, pilgrimage visa, or similar.
- "asks_for_passport": true if text asks user to send passport, KTP, personal documents, or identity documents before official process.
- "asks_upfront_payment": true if text asks payment before departure, deposit, transfer, booking fee, or upfront cost.
- "mentions_admin_fee": true if text mentions biaya administrasi, admin fee, processing fee, or similar.
- "promises_fast_process": true if text mentions proses cepat, langsung berangkat, immediate departure, fast process.
- "promises_high_salary": true if text emphasizes very high salary, gaji besar, income guarantee, unrealistic earning.
- "no_experience_required": true if text says tanpa pengalaman, no experience, no experience required.
- "no_degree_required": true if text says tanpa ijazah, no degree, no education required.
- "salary_claim_unrealistic": true if salary appears unusually high compared to typical migrant worker jobs, especially when combined with no experience or fast process.

Now extract the JSON from this input:
{{JOB_POSTING_TEXT_OR_OCR_RESULT}}
```

Bagian:

```text
{{JOB_POSTING_TEXT_OR_OCR_RESULT}}
```

adalah placeholder. FS harus menggantinya dengan teks asli dari user, OCR, caption, atau hasil pembacaan brosur.

Contoh:

```text
Now extract the JSON from this input:
DIBUTUHKAN CLEANING SERVICE MALAYSIA
Gaji 20 juta per bulan
Proses cepat, langsung berangkat
Visa turis dulu
Hubungi WhatsApp 0812xxxx
Tanpa pengalaman
```

Jika input berupa gambar, FS bisa mengirim prompt ini bersama image ke Gemini.

---

## 19. Contoh JSON Hasil Gemini yang Baik

```json
{
  "title": "Cleaning Service Malaysia",
  "location": "Malaysia, Kuala Lumpur",
  "country": "Malaysia",
  "salary_range": "Rp 20 juta per bulan",
  "description": "Gaji besar, proses cepat, visa turis dulu, segera hubungi sekarang.",
  "requirements": "Tanpa pengalaman, siap berangkat.",
  "company_profile": "",
  "employment_type": "Full-time",
  "industry": "Domestic Work",
  "benefits": "Tempat tinggal dan makan",
  "required_experience": "Not Specified",
  "required_education": "Not Specified",
  "telecommuting": 0,
  "has_company_logo": 0,
  "has_questions": 0,
  "extra": {
    "risk_signals": {
      "mentions_tourist_visa": true,
      "mentions_pilgrimage_visa": false,
      "asks_for_passport": false,
      "asks_upfront_payment": false,
      "mentions_admin_fee": false,
      "promises_fast_process": true,
      "promises_high_salary": true,
      "no_experience_required": true,
      "no_degree_required": false,
      "uses_personal_contact": true,
      "company_identity_clear": false,
      "salary_claim_unrealistic": true,
      "urgency_level": "high",
      "risk_keywords": [
        "gaji besar",
        "proses cepat",
        "visa turis",
        "hubungi sekarang",
        "tanpa pengalaman"
      ]
    }
  }
}
```

JSON ini bisa langsung dikirim ke:

```text
POST /predict
```

---

## 20. Catatan Implementasi FS

Di Express.js, flow yang disarankan:

```text
1. User upload gambar/teks lowongan.
2. Express kirim gambar/teks ke Gemini dengan prompt ekstraksi.
3. Gemini mengembalikan JSON sesuai contract.
4. Express validasi dan isi default value jika ada key hilang.
5. Express kirim JSON ke FastAPI /predict.
6. FastAPI mengembalikan final risk.
7. Express teruskan hasil ke frontend.
```

Frontend cukup menampilkan:

```text
risk_level
triggered_rules
```

Mapping UI:

```text
LOW_RISK  -> Aman
HIGH_RISK -> Berisiko
```
