# Emigria AI Model Service

Emigria AI Model Service adalah layanan backend berbasis FastAPI untuk mendeteksi potensi penipuan pada lowongan kerja luar negeri, terutama lowongan yang relevan dengan konteks Pekerja Migran Indonesia (PMI).

Service ini menerima data lowongan kerja dalam format JSON, melakukan preprocessing, menjalankan model MLP TensorFlow, lalu menggabungkan hasil model dengan PMI risk rules untuk menghasilkan skor risiko akhir.

## Tujuan Project

Project ini dibuat sebagai AI inference service untuk aplikasi Emigria. Fokus utamanya adalah membantu mengidentifikasi lowongan kerja luar negeri yang berpotensi berisiko berdasarkan:

- Pola umum lowongan kerja palsu dari model machine learning.
- Red flag khusus PMI Indonesia, seperti visa turis, biaya administrasi di awal, permintaan paspor, proses keberangkatan terlalu cepat, kontak pribadi, dan identitas perusahaan yang tidak jelas.
- Output yang bisa dipakai backend atau frontend untuk menampilkan status risiko dan alasan deteksi.

Service ini tidak menggantikan verifikasi resmi dari lembaga berwenang. Hasil prediksi sebaiknya digunakan sebagai sistem pendukung keputusan atau lapisan screening awal.

## Ringkasan Cara Kerja

Alur inference:

```text
JSON lowongan kerja
-> FastAPI
-> preprocessing dan feature engineering
-> MLP TensorFlow fraud probability
-> PMI rule-based risk scoring
-> final risk score
-> response risiko ke backend/frontend
```

Secara umum, AI service menggabungkan dua pendekatan:

1. MLP model
   Model neural network TensorFlow yang mempelajari pola lowongan palsu dari dataset training.

2. PMI rules
   Rule layer khusus untuk menangkap red flag lokal yang sering muncul pada lowongan PMI Indonesia dan belum tentu kuat terwakili di dataset training.

Formula hybrid yang digunakan:

```text
final_score = (0.3 * ml_score) + (0.7 * pmi_normalized_score)
```

Selain itu, terdapat safety logic:

- Jika MLP melewati threshold model, skor akhir minimal mengikuti skor MLP.
- Jika dua atau lebih hard-stop red flag kritis muncul bersamaan, risiko minimal dinaikkan ke level tinggi.

## Fitur Utama

- FastAPI inference service.
- Endpoint health check.
- Endpoint prediksi satu lowongan.
- Endpoint prediksi batch.
- Model MLP TensorFlow dengan artifact tersimpan.
- Preprocessing yang mirror pipeline training.
- TF-IDF text features.
- Target encoding untuk fitur kategori.
- StandardScaler untuk fitur numerik.
- PMI domain rules untuk konteks lowongan migran Indonesia.
- Dockerfile untuk deployment container.
- Docker Compose untuk menjalankan service lokal.
- Render Blueprint untuk deployment ke Render.

## Tech Stack

- Python 3.11
- FastAPI
- Uvicorn
- TensorFlow / Keras
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Category Encoders
- Docker
- Render

## Struktur Folder

```text
.
|-- api/
|   |-- app.py
|   |-- preprocessing.py
|   |-- pmi_rules.py
|   |-- requirements.txt
|   |-- Dockerfile
|   |-- README.md
|   `-- models/
|       |-- emigria_mlp_model.keras
|       |-- target_encoder.pkl
|       |-- tfidf_vectorizer.pkl
|       |-- standard_scaler.pkl
|       |-- country_salary_avg.pkl
|       |-- feature_columns.json
|       |-- threshold.txt
|       |-- preprocessing_config.json
|       |-- metrics.json
|       `-- threshold_tuning.csv
|-- docs/
|   |-- AI_MODELING_CONTEXT_FOR_FS.md
|   |-- ARSITEKTUR_EMIGRIA.md
|   |-- FASTAPI_DOCKER_GUIDE_FS.md
|   `-- HYBRID_MLP_PMI_UPDATE_FOR_FS.md
|-- docker-compose.yml
|-- render.yaml
|-- .dockerignore
|-- .gitignore
`-- README.md
```
## AI Development
https://colab.research.google.com/drive/1MFuZI4uDeJPVSPKda0iuZZAgfU_bUZtz#scrollTo=dOjzjrfv_3Cj



## Artifact Model

Artifact model berada di:

```text
api/models/
```

File yang dibutuhkan agar service siap melakukan prediksi:

```text
emigria_mlp_model.keras
target_encoder.pkl
tfidf_vectorizer.pkl
standard_scaler.pkl
country_salary_avg.pkl
feature_columns.json
threshold.txt
preprocessing_config.json
```

File tambahan untuk dokumentasi dan evaluasi:

```text
metrics.json
threshold_tuning.csv
```

Jika artifact belum lengkap, aplikasi tetap bisa start, tetapi endpoint `/health` akan mengembalikan status `missing_artifacts`. Endpoint prediksi akan mengembalikan error `503` sampai artifact lengkap tersedia.

## Metrik Model

Berdasarkan `api/models/metrics.json`, performa model pada test set dengan threshold `0.5`:

```text
Precision fraud : 0.6919
Recall fraud    : 0.7301
F1 fraud        : 0.7104
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

Interpretasi singkat:

- Dari 163 data fraud pada test set, model menangkap 119.
- Terdapat 44 data fraud yang tidak terdeteksi oleh model murni.
- Ada 53 false alarm pada data non-fraud.

Karena dataset training belum sepenuhnya spesifik untuk konteks PMI Indonesia, service ini menggunakan rule layer tambahan agar lebih sensitif terhadap red flag lokal.

## Instalasi Lokal Tanpa Docker

Pastikan Python 3.11 sudah tersedia.

Install dependency:

```bash
pip install -r api/requirements.txt
```

Jalankan server:

```bash
uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
```

Service akan tersedia di:

```text
http://localhost:8000
```

Health check:

```bash
curl http://localhost:8000/health
```

Jika semua artifact tersedia, response akan berisi:

```json
{
  "status": "ready",
  "model_dir": "...",
  "missing_artifacts": [],
  "threshold": 0.5
}
```

## Menjalankan Dengan Docker Compose

Dari root project:

```bash
docker compose up --build
```

Service akan berjalan di:

```text
http://localhost:8000
```

Docker Compose akan mount folder model sebagai read-only:

```text
./api/models:/app/api/models:ro
```

## Deployment ke Render

Project ini sudah memiliki `render.yaml` untuk deployment sebagai Docker Web Service.

Konfigurasi utama:

```text
runtime: docker
dockerfilePath: ./api/Dockerfile
dockerContext: .
healthCheckPath: /health
```

Dockerfile menggunakan image:

```text
python:3.11.15-slim-bookworm
```

Server membaca environment variable `PORT` dari Render dan fallback ke port `8000` untuk lokal.

## Endpoint API

### GET `/health`

Mengecek status service dan kelengkapan artifact model.

Contoh:

```bash
curl http://localhost:8000/health
```

### POST `/predict`

Melakukan prediksi untuk satu lowongan kerja.

Contoh request:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
    "has_questions": 0
  }'
```

### POST `/predict-batch`

Melakukan prediksi untuk banyak lowongan sekaligus.

Contoh struktur request:

```json
{
  "records": [
    {
      "title": "Cleaning Service Malaysia",
      "location": "Malaysia, Kuala Lumpur",
      "country": "Malaysia",
      "salary_range": "Rp 20 juta per bulan",
      "description": "Gaji besar, proses cepat, visa turis dulu.",
      "requirements": "Tanpa pengalaman.",
      "company_profile": "",
      "employment_type": "Full-time",
      "industry": "Domestic Work",
      "benefits": "",
      "required_experience": "Not Specified",
      "required_education": "Not Specified",
      "telecommuting": 0,
      "has_company_logo": 0,
      "has_questions": 0
    }
  ]
}
```

## Input Contract

Field utama yang direkomendasikan untuk selalu dikirim:

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

Default value:

```text
Text kosong         : ""
employment_type     : "Unknown"
industry            : "Unknown"
required_experience : "Not Specified"
required_education  : "Not Specified"
telecommuting       : 0
has_company_logo    : 0
has_questions       : 0
```

Binary value:

```text
0 = tidak / tidak ditemukan
1 = ya / ditemukan
```

Field tambahan bisa dikirim melalui `extra`. Service akan menggabungkan isi `extra` ke record prediksi.

## Contoh Response

Response `/predict` berisi skor dari model, skor rule PMI, dan hasil akhir.

Contoh:

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

Field yang paling penting untuk frontend:

```text
risk_level
final_risk_percentage
ml_fraud_percentage
pmi_risk_percentage
triggered_rules
fraud_prediction
```

Risk level internal:

```text
LOW_RISK  : final_risk_score < 0.40
REVIEW    : 0.40 <= final_risk_score < 0.60
HIGH_RISK : final_risk_score >= 0.60
```

## Red Flag PMI yang Dideteksi

PMI rule layer membaca teks dari field:

```text
title
description
requirements
benefits
company_profile
salary_range
```

Contoh red flag:

- Penggunaan visa turis, visa wisata, visa ziarah, visa umroh, atau visa on arrival.
- Permintaan mengirim paspor, dokumen asli, KTP, atau data sensitif.
- Biaya administrasi, biaya keberangkatan, uang muka, transfer di awal, atau upfront payment.
- Proses cepat, langsung berangkat, berangkat hari ini, atau proses kilat.
- Janji gaji besar, gaji tidak masuk akal, bonus besar, atau guaranteed income.
- Tanpa pengalaman, tanpa ijazah, tanpa interview, atau langsung diterima.
- Kontak pribadi melalui WhatsApp, Telegram, email gratis, atau nomor personal.
- Identitas perusahaan tidak jelas, company profile kosong, tidak ada logo perusahaan, atau tidak ada screening questions.
- Sinyal pekerjaan berbahaya seperti scam center, casino online, judi online, atau operator situs ilegal.
- Negara atau wilayah yang diberi bobot risiko lebih tinggi untuk konteks tertentu, seperti Kamboja, Myanmar, Laos, dan Macau.

Rule layer juga bisa mengurangi skor jika ada sinyal resmi, seperti BP2MI, P3MI, SIPMI, SIPP TKIS, izin operasional, kontrak kerja resmi, atau tidak ada biaya penempatan.

## Integrasi Dengan Backend Utama

Contoh alur integrasi yang disarankan:

```text
1. User mengirim teks, gambar, atau brosur lowongan.
2. Backend utama melakukan OCR atau ekstraksi informasi.
3. Gemini atau extractor lain mengubah input menjadi JSON lowongan.
4. Backend utama memvalidasi field dan mengisi default value.
5. Backend utama mengirim JSON ke FastAPI `/predict`.
6. FastAPI mengembalikan risk score dan triggered rules.
7. Backend/frontend menampilkan hasil analisis ke user.
```

Gemini tidak menjadi penentu akhir fraud. Gemini hanya disarankan sebagai extraction layer untuk mengubah gambar, brosur, atau teks mentah menjadi JSON terstruktur.

## Catatan Keterbatasan

- Dataset training model belum sepenuhnya spesifik untuk lowongan PMI Indonesia.
- MLP bisa kurang sensitif terhadap pola lokal yang tidak ada di dataset training.
- PMI rules membantu menangkap red flag lokal, tetapi tetap berbasis keyword dan heuristik.
- Hasil prediksi perlu diperlakukan sebagai indikasi risiko, bukan keputusan legal final.
- Untuk peningkatan jangka panjang, dibutuhkan dataset lowongan PMI Indonesia yang lebih representatif dan sudah diberi label.

## Dokumentasi Tambahan

Dokumen teknis lain tersedia di folder `docs/`:

```text
docs/AI_MODELING_CONTEXT_FOR_FS.md
docs/HYBRID_MLP_PMI_UPDATE_FOR_FS.md
docs/FASTAPI_DOCKER_GUIDE_FS.md
docs/ARSITEKTUR_EMIGRIA.md
```

Untuk dokumentasi khusus service FastAPI, lihat:

```text
api/README.md
```

## Status Project

Project ini sudah siap digunakan sebagai AI inference service selama artifact model tersedia lengkap di `api/models/`.

Mode yang tersedia:

- Local Python dengan Uvicorn.
- Docker Compose.
- Docker deployment ke Render.

## Lisensi

Lisensi belum ditentukan. Tambahkan file `LICENSE` jika project akan dipublikasikan atau digunakan oleh pihak lain.
