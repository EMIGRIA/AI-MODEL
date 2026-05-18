# Panduan Docker Compose FastAPI AI Service untuk Tim FS

Dokumen ini menjelaskan cara menjalankan AI microservice Emigria secara lokal dengan Docker Compose.

Service ini berisi:

```text
FastAPI
-> preprocessing job posting
-> MLP TensorFlow fraud probability
-> PMI rule-based risk layer
-> final LOW_RISK / HIGH_RISK
```

---

## 1. Struktur Folder yang Dibutuhkan

Pastikan struktur project seperti ini:

```text
ai model/
├── api/
│   ├── app.py
│   ├── preprocessing.py
│   ├── pmi_rules.py
│   ├── requirements.txt
│   ├── Dockerfile
│   └── models/
│       ├── emigria_mlp_model.keras
│       ├── target_encoder.pkl
│       ├── tfidf_vectorizer.pkl
│       ├── standard_scaler.pkl
│       ├── country_salary_avg.pkl
│       ├── feature_columns.json
│       ├── threshold.txt
│       ├── preprocessing_config.json
│       ├── metrics.json
│       └── threshold_tuning.csv
├── docker-compose.yml
└── .dockerignore
```

File yang wajib ada di `api/models/`:

```text
emigria_mlp_model.keras
target_encoder.pkl
tfidf_vectorizer.pkl
standard_scaler.pkl
country_salary_avg.pkl
threshold.txt
```

File tambahan yang direkomendasikan:

```text
feature_columns.json
preprocessing_config.json
metrics.json
threshold_tuning.csv
```

---

## 2. Jalankan Dengan Docker Compose

Dari root project, yaitu folder yang punya `docker-compose.yml`, jalankan:

```bash
docker compose up --build
```

Jika berhasil, FastAPI tersedia di:

```text
http://localhost:8000
```

Dokumentasi Swagger tersedia di:

```text
http://localhost:8000/docs
```

---

## 3. Health Check

Cek status service:

```bash
curl http://localhost:8000/health
```

Jika artifact model lengkap, response harus seperti:

```json
{
  "status": "ready",
  "model_dir": "/app/api/models",
  "missing_artifacts": [],
  "threshold": 0.5
}
```

Jika ada file model yang belum dimasukkan, response akan menunjukkan:

```json
{
  "status": "missing_artifacts",
  "missing_artifacts": [
    "models/emigria_mlp_model.keras"
  ]
}
```

Solusi:

```text
Masukkan file yang hilang ke api/models/, lalu restart container.
```

---

## 4. Endpoint Predict

Endpoint:

```text
POST http://localhost:8000/predict
```

Contoh request:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Cleaning Service Malaysia",
    "location": "Malaysia, Kuala Lumpur",
    "country": "Malaysia",
    "salary_range": "1500-2000",
    "description": "Gaji besar, proses cepat, visa turis, segera hubungi sekarang!",
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
    "hubungi sekarang",
    "company_profile_empty",
    "no_company_logo",
    "no_screening_questions"
  ],
  "fraud_prediction": 1,
  "risk_level": "HIGH_RISK",
  "threshold": 0.5,
  "pmi_rule_threshold": 4
}
```

---

## 5. Penjelasan Response

Field penting:

```text
ml_fraud_probability
```

Probabilitas fraud dari model MLP TensorFlow.

```text
ml_fraud_prediction
```

Prediksi biner dari MLP saja.

```text
pmi_rule_score
```

Skor rule-based untuk red flag PMI, misalnya visa turis, biaya administrasi, kirim paspor, proses cepat, dan profil perusahaan kosong.

```text
triggered_rules
```

Daftar alasan kenapa rule layer menandai lowongan sebagai berisiko.

```text
fraud_prediction
```

Prediksi final setelah MLP digabung dengan PMI rule layer.

```text
risk_level
```

Label final untuk frontend:

```text
LOW_RISK
HIGH_RISK
```

Tidak ada `MEDIUM_RISK` supaya tampilan user tidak membingungkan.

---

## 6. Logic Final Risk

FastAPI menggunakan hybrid logic:

```text
Jika MLP probability >= threshold
atau PMI rule score >= 4
maka HIGH_RISK.

Selain itu LOW_RISK.
```

Dengan kata lain:

```text
MLP menangkap pola fraud umum dari dataset.
PMI rule layer menangkap red flag spesifik lowongan PMI Indonesia.
```

---

## 7. Request dari Express.js

Express.js bisa mengirim JSON hasil ekstraksi Gemini langsung ke endpoint `/predict`.

Minimal field yang direkomendasikan:

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

Jika Gemini tidak menemukan sebuah field, kirim default:

```text
String kosong       -> ""
Kategori tidak tahu -> "Unknown" atau "Not Specified"
Binary             -> 0
```

---

## 8. Contoh Fetch dari Express.js

```js
const response = await fetch("http://localhost:8000/predict", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
  },
  body: JSON.stringify(extractedJobPosting),
});

const result = await response.json();
console.log(result);
```

Untuk Docker network antar-service, jika Express juga berjalan di Docker Compose yang sama, URL bisa diganti menjadi:

```text
http://emigria-ai-service:8000/predict
```

---

## 9. Troubleshooting

### Docker command tidak dikenali

Pastikan Docker Desktop sudah terinstall dan berjalan.

```bash
docker --version
docker compose version
```

### Status health `missing_artifacts`

Penyebab:

```text
File model belum ada di api/models/
```

Solusi:

```text
Extract artifact zip dari Colab ke api/models/
Restart container.
```

### TensorFlow lama saat build

Jika build lama, ini normal karena TensorFlow cukup besar.

Gunakan:

```bash
docker compose up --build
```

Untuk run berikutnya:

```bash
docker compose up
```

### Port 8000 sudah dipakai

Ubah mapping port di `docker-compose.yml`:

```yaml
ports:
  - "8001:8000"
```

Lalu akses:

```text
http://localhost:8001
```

### Response selalu HIGH_RISK untuk lowongan Indonesia

Ini bisa terjadi jika PMI rule layer mendeteksi red flag seperti:

```text
visa turis
biaya administrasi
kirim paspor
proses cepat
company profile kosong
```

Cek field:

```text
triggered_rules
```

untuk melihat alasan final risk.

---

## 10. Ringkasan Untuk FS

Yang perlu dilakukan FS:

```text
1. Pastikan artifact model ada di api/models/
2. Jalankan docker compose up --build
3. Cek /health
4. Kirim JSON ekstraksi Gemini ke /predict
5. Pakai risk_level dan triggered_rules untuk tampilan frontend
```
