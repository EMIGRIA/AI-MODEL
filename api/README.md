# Emigria FastAPI AI Service

FastAPI service ini melayani inference model MLP Emigria dan menambahkan PMI risk rule layer.

## Struktur Artifact

Taruh artifact hasil Colab di folder:

```text
api/models/
```

File yang dibutuhkan:

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

## Run Lokal Tanpa Docker

```bash
pip install -r api/requirements.txt
uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
```

## Run Dengan Docker Compose

Dari root project:

```bash
docker compose up --build
```

FastAPI akan tersedia di:

```text
http://localhost:8000
```

Health check:

```bash
curl http://localhost:8000/health
```

Jika artifact lengkap, response `status` akan bernilai `ready`.

## Deploy ke Render

Cara paling aman untuk project ini adalah deploy sebagai Docker Web Service.
Versi Python dikunci di `api/Dockerfile`:

```text
python:3.11.15-slim-bookworm
```

Jika memakai Blueprint, gunakan `render.yaml` di root repo. Render akan memakai:

```text
runtime: docker
dockerfilePath: ./api/Dockerfile
dockerContext: .
healthCheckPath: /health
```

Jika setup manual dari dashboard Render:

```text
Language/Runtime: Docker
Dockerfile Path: ./api/Dockerfile
Docker Build Context Directory: .
Health Check Path: /health
```

Dockerfile sudah membaca env `PORT` dari Render dan fallback ke `8000` untuk lokal.

## Endpoint

```text
GET /health
POST /predict
POST /predict-batch
```

`POST /predict` menerima JSON hasil ekstraksi Gemini/Express.

Response menggabungkan:

```text
MLP fraud probability + PMI rule score = final fraud prediction
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
