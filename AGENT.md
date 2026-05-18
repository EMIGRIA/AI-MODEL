# Emigria AI Engineer Guide - MLP Mirror Pipeline DS

Dokumen ini adalah panduan kerja final untuk membangun model **MLP TensorFlow** pada project capstone **Emigria**, dengan cara **mirror pipeline preprocessing notebook final tim Data Science**, lalu mengganti classifier RF/XGBoost menjadi MLP.

Fokus utama:

```text
fake_job_postings.csv
-> cleaning sesuai notebook DS
-> feature engineering sesuai notebook DS
-> train/validation/test split stratified
-> target encoding leakage-free
-> TF-IDF description
-> StandardScaler
-> MLP TensorFlow
-> evaluasi fraud detection
-> threshold tuning
-> export model dan artifact preprocessing
-> siap dipakai FastAPI
```

---

## 1. Keputusan Pipeline

Pipeline final AI Engineer **tidak memakai `main_data.csv` sebagai sumber training utama**.

Dataset utama yang dipakai:

```text
fake_job_postings.csv
```

Alasan:

1. `main_data.csv` dari tim DS adalah output akhir untuk dashboard, bukan dataset mentah sebelum preprocessing.
2. Banyak fitur di `main_data.csv` sudah ditimpa dengan versi hasil `StandardScaler`.
3. `main_data.csv` sudah mengandung output model DS:
   - `fraud_prediction`
   - `fraud_probability`
   - `risk_level`
4. Notebook DS membangun `main_data.csv` dari `fake_job_postings.csv`.
5. FastAPI nanti menerima JSON mentah hasil ekstraksi Gemini, sehingga lebih aman jika pipeline training juga dimulai dari data mentah.

Kesimpulan:

```text
Untuk baseline dari dataset siap pakai -> main_data.csv boleh dipakai.
Untuk pipeline production-like dan FastAPI -> mirror notebook DS dari fake_job_postings.csv.
```

Dokumen ini mengikuti pilihan kedua.

---

## 2. Peran Komponen Sistem Emigria

Arsitektur project:

```text
Frontend React/Vite
-> Express.js orchestrator
-> Gemini API untuk ekstraksi brosur menjadi JSON
-> FastAPI AI microservice
-> MLP TensorFlow untuk fraud detection
```

AI Engineer bertanggung jawab pada:

1. Membuat pipeline preprocessing dan feature engineering.
2. Melatih model MLP.
3. Mengevaluasi model dengan metrik fraud detection.
4. Menyimpan model dan artifact preprocessing.
5. Menyediakan fungsi inference yang bisa dipakai FastAPI.

---

## 3. Dataset yang Dipakai

File input:

```text
data/fake_job_postings.csv
```

Di Google Colab, file bisa di-upload sebagai:

```text
/content/fake_job_postings.csv
/content/data/fake_job_postings.csv
```

Target label:

```text
fraudulent
```

Makna label:

```text
0 = non-fraud / legitimate / aman
1 = fraud / berisiko penipuan
```

Masalah utama dataset:

```text
class imbalance
```

Biasanya kelas fraud jauh lebih sedikit daripada non-fraud. Karena itu, accuracy tidak boleh menjadi metrik utama.

---

## 4. Kolom Raw yang Dibutuhkan dari Dataset

Pipeline mirror DS mengambil kolom berikut dari `fake_job_postings.csv`:

```python
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
```

Untuk inference FastAPI, request JSON dari Gemini sebaiknya mengikuti field yang mirip:

```json
{
  "title": "Cleaning Service",
  "location": "Malaysia, Kuala Lumpur",
  "country": "Malaysia",
  "salary_range": "1500-2000",
  "description": "Gaji besar, proses cepat...",
  "requirements": "",
  "company_profile": "",
  "employment_type": "Full-time",
  "industry": "Domestic Work",
  "benefits": "Tempat tinggal",
  "required_experience": "Not Specified",
  "required_education": "Not Specified",
  "telecommuting": 0,
  "has_company_logo": 0,
  "has_questions": 0
}
```

Jika field tidak ditemukan oleh Gemini, isi default:

```text
string -> ""
category -> "Unknown" atau "Not Specified"
binary -> 0
```

---

## 5. Cleaning Data

### 5.1 Parse Salary

Fungsi `parse_salary()` mengubah `salary_range` menjadi:

```text
salary_min
salary_max
```

Contoh:

```text
"30000-50000" -> 30000, 50000
"30k-50k"    -> 30000, 50000
missing      -> NaN, NaN
```

Jika angka terlalu kecil seperti `10-20`, notebook DS mengasumsikan itu hourly rate dan mengalikan dengan `2080`.

### 5.2 Extract Country

Notebook DS memakai `location` untuk mengambil kode negara.

Contoh:

```text
"US, NY, New York" -> "United States"
"MY, Kuala Lumpur" -> "Malaysia"
```

Mapping kode negara disimpan dalam `COUNTRY_MAP`.

Untuk FastAPI, jika Gemini sudah memberikan field `country`, field itu bisa dipakai langsung. Jika tidak ada, sistem fallback ke parsing `location`.

### 5.3 Fill Missing Values

Text columns:

```python
["description", "requirements", "company_profile", "benefits"]
```

Default:

```text
""
```

Categorical columns:

```python
["employment_type", "industry", "title"]
```

Default:

```text
"Unknown"
```

Education/experience:

```python
["required_experience", "required_education"]
```

Default:

```text
"Not Specified"
```

Binary columns:

```python
["telecommuting", "has_company_logo", "has_questions"]
```

Default:

```text
0
```

### 5.4 Deduplication

Notebook DS menghapus duplikasi berdasarkan:

```python
["title", "description", "location"]
```

Untuk mirror notebook DS, training script juga melakukan deduplication.

Catatan:

```text
Dedup boleh mengurangi jumlah data fraud.
Karena ini mengikuti notebook final DS, tetap dilakukan untuk konsistensi.
```

---

## 6. Feature Engineering Dasar

Fitur gaji:

```python
salary_mid = (salary_min + salary_max) / 2
salary_spread = salary_max - salary_min
salary_spread_ratio = salary_spread / salary_mid
has_salary = salary_min.notna()
```

Fitur teks:

```python
title_length = len(title)
desc_length = len(description)
req_length = len(requirements)
has_company_profile = company_profile length > 0
```

Fitur fraud signal:

```python
scam_keyword_score
has_email_in_desc
is_free_email
exclamation_count
is_indonesian_posting
```

`scam_keyword_score` memakai keyword Bahasa Indonesia dan Bahasa Inggris, dengan bobot:

```text
high risk keyword   -> +3
medium risk keyword -> +2
low risk keyword    -> +1
```

Contoh high risk keyword:

```text
visa turis
visa ziarah
kirim paspor
bayar di muka
biaya administrasi
free visa
send passport
upfront payment
```

---

## 7. Split Data

Pipeline final memakai tiga split:

```text
train      -> fit preprocessing dan model
validation -> early stopping dan threshold tuning
test       -> final evaluation
```

Semua split wajib stratified:

```python
train_test_split(..., stratify=y)
```

Alasan:

```text
Fraud class sangat kecil, jadi distribusi label harus dijaga di semua split.
```

---

## 8. Feature Engineering Leakage-Free Setelah Split

Bagian ini adalah inti mirror notebook DS.

Fungsi produksi yang dipakai:

```text
fit_preprocessing_artifacts(X_train_raw, y_train)
transform_features(X_raw, artifacts)
```

Kenapa dipisah?

```text
Artifact seperti TargetEncoder, TF-IDF, country salary average, dan scaler hanya boleh fit di training data.
Validation, test, dan inference hanya boleh transform.
```

### 8.1 Target Encoding

Notebook DS memakai target encoding untuk:

```python
["country", "employment_type", "industry"]
```

Output fitur:

```python
country_fraud_rate
country_safety_score = 1 - country_fraud_rate
emp_type_fraud_rate
industry_fraud_rate
```

Penting:

```text
TargetEncoder hanya fit pada train set.
Jangan fit pada seluruh dataset karena itu menyebabkan data leakage.
```

### 8.2 Country Average Salary

Hitung median `salary_mid` dari data training yang legitimate saja:

```python
legit_train = train data where fraudulent == 0 and salary_mid notna
country_avg_salary = median salary_mid per country
global_salary_median = median salary_mid legitimate train
```

Output fitur:

```python
country_avg_salary
salary_vs_country_avg = salary_mid / country_avg_salary
```

Penting:

```text
Jangan hitung country_avg_salary dari seluruh dataset.
Hanya train set yang boleh dipakai.
```

### 8.3 TF-IDF Description

Notebook DS memakai:

```python
TfidfVectorizer(
    max_features=100,
    stop_words="english",
    sublinear_tf=True,
    ngram_range=(1, 2)
)
```

Output fitur:

```text
tfidf_0
tfidf_1
...
tfidf_99
```

Penting:

```text
TF-IDF vectorizer fit hanya pada description train set.
Validation, test, dan FastAPI hanya transform.
```

---

## 9. Final Feature List

Fitur non-TFIDF:

```python
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
```

TF-IDF features:

```python
tfidf_0 ... tfidf_99
```

Total fitur:

```text
22 fitur dasar + 100 fitur TF-IDF = 122 fitur
```

---

## 10. Scaling

Setelah semua fitur numerik dibuat, gunakan:

```python
StandardScaler()
```

Rules:

```text
scaler.fit hanya pada X_train_unscaled
scaler.transform pada X_val_unscaled, X_test_unscaled, dan inference
```

Hasil scaler disimpan:

```text
models/standard_scaler.pkl
```

---

## 11. Model MLP TensorFlow

Baseline arsitektur:

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

Requirement Main Quest yang dipenuhi:

```text
Deep Learning memakai TensorFlow Functional API.
```

Loss:

```text
binary_crossentropy
```

Class imbalance:

```text
class_weight="balanced"
```

Metrics TensorFlow:

```python
accuracy
precision
recall
roc_auc
pr_auc
```

Catatan:

```text
Accuracy boleh dilaporkan, tetapi bukan metrik utama.
Untuk fraud detection, fokus ke recall fraud, precision fraud, F1 fraud, PR-AUC, ROC-AUC, confusion matrix.
```

---

## 12. Callback Training

Gunakan:

```python
EarlyStopping(monitor="val_pr_auc", mode="max", restore_best_weights=True)
ReduceLROnPlateau(monitor="val_pr_auc", mode="max")
```

Selain callback built-in, script wajib menambahkan komponen kustom:

```python
FraudMetricsCallback
```

Fungsi custom callback:

```text
1. Menghitung precision fraud pada validation set per epoch.
2. Menghitung recall fraud pada validation set per epoch.
3. Menghitung F1 fraud pada validation set per epoch.
4. Menyimpan best validation F1 fraud dan epoch terbaik.
```

Ini memenuhi Main Quest:

```text
Mengimplementasikan setidaknya satu komponen kustom lanjutan:
Custom Callback.
```

Alasan:

```text
PR-AUC lebih informatif untuk dataset imbalanced daripada accuracy.
```

---

## 13. Evaluasi Model

Evaluasi wajib mencakup:

```text
classification report
confusion matrix
ROC-AUC
PR-AUC
precision fraud
recall fraud
F1-score fraud
```

Confusion matrix:

```text
[[TN FP]
 [FN TP]]
```

Interpretasi penting:

```text
FN = fraud diprediksi aman.
Untuk Emigria, FN berbahaya karena calon PMI bisa lolos dari peringatan.
```

---

## 14. Threshold Tuning

Default threshold:

```text
0.5
```

Namun untuk fraud detection, threshold sering perlu diturunkan agar recall fraud meningkat.

Threshold tuning dilakukan pada validation set:

```text
threshold 0.05 sampai 0.95
```

Pilih dua kandidat:

1. `best_f1_threshold`
   - threshold dengan F1 fraud terbaik di validation set.
2. `recall_aware_threshold`
   - threshold dengan recall fraud minimal 0.80 jika memungkinkan.

Threshold final untuk MVP Emigria:

```text
0.50
```

Alasan:

```text
Threshold 0.50 menjadi kompromi antara recall fraud yang cukup tinggi dan precision yang masih terkendali.
Best F1 threshold tetap dilaporkan sebagai analisis, tetapi tidak otomatis dijadikan threshold produksi.
Recall-aware threshold juga dilaporkan, tetapi tidak dipakai default jika terlalu agresif dan menghasilkan terlalu banyak false alarm.
```

Final evaluation tetap dilakukan di test set, memakai threshold yang dipilih dari validation set.

---

## 15. Artifact yang Harus Disimpan

Folder output:

```text
models/
```

Artifact utama:

```text
models/emigria_mlp_model.keras
models/target_encoder.pkl
models/tfidf_vectorizer.pkl
models/standard_scaler.pkl
models/country_salary_avg.pkl
models/feature_columns.json
models/threshold.txt
models/threshold_tuning.csv
models/metrics.json
models/preprocessing_config.json
```

Export `.keras` memenuhi Main Quest:

```text
Menyimpan dan mengekspor model TensorFlow siap produksi.
```

Artifact ini dibutuhkan FastAPI untuk melakukan transformasi input Gemini menjadi fitur MLP.

---

## 16. Inference FastAPI

Alur inference:

```text
Request JSON dari Express/Gemini
-> clean raw fields
-> parse salary
-> extract/fill country
-> feature engineering dasar
-> target encoder transform
-> country salary avg lookup
-> TF-IDF transform description
-> susun feature columns
-> scaler transform
-> MLP predict probability
-> threshold
-> risk level
-> response ke Express.js
```

Response ideal:

```json
{
  "fraud_probability": 0.82,
  "fraud_prediction": 1,
  "risk_level": "HIGH_RISK",
  "threshold": 0.43
}
```

Script training juga wajib memiliki smoke test inference sederhana:

```text
1. Load model .keras.
2. Load target encoder, TF-IDF vectorizer, scaler, dan country salary artifact.
3. Ambil satu raw sample.
4. Jalankan preprocessing.
5. Prediksi fraud probability.
6. Bentuk response mirip API.
```

Ini memenuhi Main Quest:

```text
Membuat kode sederhana untuk proses inference model.
```

Risk level sederhana:

```python
if probability >= threshold:
    "HIGH_RISK"
else:
    "LOW_RISK"
```

Untuk tampilan user, Emigria hanya memakai dua status:

```text
LOW_RISK  -> aman
HIGH_RISK -> berisiko
```

Status medium tidak ditampilkan agar pengguna tidak bingung.

---

## 17. Kesalahan yang Harus Dihindari

Jangan melakukan ini:

```text
1. Fit TargetEncoder pada seluruh dataset.
2. Fit TF-IDF pada seluruh dataset.
3. Fit scaler pada seluruh dataset.
4. Tuning threshold di test set.
5. Menilai model hanya dari accuracy.
6. Memakai fraud_prediction/fraud_probability/risk_level sebagai fitur.
7. Melakukan double scaling terhadap fitur yang sudah scaled tanpa memahami asal pipeline.
8. Menggunakan main_data.csv untuk inference raw Gemini tanpa preprocessing DS asli.
```

---

## 18. Perbedaan Dengan Script Lama `emigria_mlp_final.py`

Script lama adalah eksperimen dan tidak dijadikan pipeline final.

Masalah script lama:

```text
1. Preprocessing belum sepenuhnya mirror notebook final DS.
2. Fitur lebih sedikit daripada notebook DS.
3. Inference demo berpotensi melakukan double scaling.
4. Artifact belum disusun agar mudah dipakai FastAPI.
5. Validasi dan threshold tuning belum dipisahkan sebersih train/val/test.
```

Pipeline baru harus memakai script:

```text
train_mlp_mirror_ds_colab.py
```

---

## 19. Checklist Training

Sebelum training:

```text
[ ] fake_job_postings.csv tersedia
[ ] target fraudulent tersedia
[ ] class imbalance dicek
[ ] cleaning sesuai notebook DS
[ ] feature engineering dasar sesuai notebook DS
[ ] split stratified train/val/test
[ ] target encoder fit hanya train
[ ] TF-IDF fit hanya train
[ ] country salary avg hitung hanya train legitimate
[ ] scaler fit hanya train
```

Saat training:

```text
[ ] MLP Functional API
[ ] Custom callback FraudMetricsCallback aktif
[ ] binary_crossentropy
[ ] class_weight aktif
[ ] monitor val_pr_auc
[ ] early stopping aktif
```

Setelah training:

```text
[ ] classification report validation/test
[ ] confusion matrix test
[ ] ROC-AUC test
[ ] PR-AUC test
[ ] threshold tuning validation
[ ] final evaluation test threshold terpilih
[ ] model .keras disimpan
[ ] preprocessing artifacts disimpan
[ ] smoke test inference berhasil
```

---

## 20. Kesimpulan

Untuk Emigria, pipeline MLP yang paling kuat secara engineering adalah:

```text
fake_job_postings.csv
-> mirror preprocessing notebook DS
-> ganti classifier RF/XGB menjadi MLP
-> simpan semua preprocessing artifacts
-> FastAPI memakai preprocessing yang sama saat inference
```

Dengan alur ini, model tidak hanya valid untuk eksperimen, tetapi juga lebih masuk akal untuk integrasi backend karena input FastAPI berasal dari JSON mentah hasil ekstraksi Gemini.
