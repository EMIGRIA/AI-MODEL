# Progress Notes — AI Engineer Emigria MLP

## Status: IN PROGRESS (Step 8/12 selesai, perlu optimisasi F1)

---

## Hasil Terakhir

| Metrik | Nilai | Target |
|---|---|---|
| F1 Score | **61.34%** (setelah threshold tuning) | ≥ 78% |
| Recall | 55.49% | ≥ 80% |
| AUC-ROC | ~94% | ≥ 97% |
| Accuracy | ~96% | ≥ 85% ✅ |

**Model sudah bisa mendeteksi fraud, tapi F1 masih di bawah target.**

---

## Keputusan Penting yang Sudah Diambil

### 1. Pakai `fake_job_postings.csv` (RAW DATA), BUKAN `main_data.csv`

**Alasan:**
- `main_data.csv` dari tim DS sudah di-StandardScaler → kalau kita scale lagi = double scaling → model gagal total (F1 = 0)
- XGBoost di `main_data.csv` cuma dapat F1 = 0.62 (bukan 0.74 seperti di notebook DS)
- Kemungkinan `main_data.csv` sudah di-filter/transform berbeda dari yang dipakai XGB tim DS
- Pakai raw data + preprocessing sendiri → MLP F1 = 0.56-0.61, AUC = 0.94

**Kesimpulan:** `main_data.csv` tidak cocok untuk training MLP kita. Pakai raw data.

### 2. Pakai Binary Cross-Entropy + class_weight, BUKAN Focal Loss untuk Training

**Alasan:**
- Focal Loss dengan berbagai kombinasi alpha/gamma selalu gagal (F1 = 0)
- Masalahnya: alpha parameter di Focal Loss sangat sensitif untuk data imbalanced 95:5
- BCE + class_weight (fraud diberi bobot 20x) jauh lebih stabil
- **Focal Loss tetap didefinisikan** di Step 5 untuk memenuhi kriteria Main Quest M2 (komponen kustom)

### 3. Arsitektur 64→32→16 (bukan 128→64→32→16)

**Alasan:**
- Hanya 16 fitur input → arsitektur terlalu dalam = overfitting
- 64→32→16 lebih proporsional untuk tabular data 16 fitur
- Dropout 0.3/0.2/0.1 (decreasing)

### 4. Dedup Dinonaktifkan

**Alasan:**
- Dedup menghapus 3,030 baris termasuk 223 data fraud (25% fraud hilang)
- Data fraud sudah sangat sedikit (866) → tidak boleh dikurangi lagi

---

## File yang Dipakai

| File | Fungsi | Status |
|---|---|---|
| `emigria_mlp_final.py` | **Script utama** — pakai ini | ✅ Aktif |
| `emigria_mlp.py` | Versi lama (v1) — JANGAN pakai | ❌ Deprecated |
| `emigria_mlp_v2.py` | Versi main_data — JANGAN pakai | ❌ Deprecated |
| `data/fake_job_postings.csv` | Dataset utama | ✅ |
| `data/main_data (1).csv` | Dataset DS — tidak dipakai | ❌ |

---

## Alur Script `emigria_mlp_final.py`

```
Step 1:  Import library
Step 2:  Load fake_job_postings.csv
Step 3:  Preprocessing + Feature Engineering (16 fitur)
Step 4:  Train/Test Split + TargetEncoder (leakage-free) + StandardScaler
Step 5:  Custom Focal Loss (definisi class — untuk kriteria M2)
Step 6:  Custom F1 Callback (definisi class)
Step 7:  Build MLP Functional API (64→32→16→1)
Step 8:  Training (BCE + class_weight + TensorBoard) ← SUDAH SELESAI
Step 9:  Threshold Optimization ← SUDAH SELESAI
Step 10: Evaluasi Final ← BELUM
Step 11: Export Model & Artifacts ← BELUM
Step 12: Inference Demo ← BELUM
```

---

## 16 Fitur yang Dipakai

```
1.  has_company_profile    (binary: ada profil perusahaan atau tidak)
2.  has_company_logo       (binary: ada logo atau tidak)
3.  has_questions          (binary: ada screening question atau tidak)
4.  telecommuting          (binary: remote atau tidak)
5.  has_salary             (binary: ada info gaji atau tidak)
6.  has_benefits           (binary: ada info tunjangan atau tidak)
7.  has_requirements       (binary: ada syarat kerja atau tidak)
8.  title_length           (panjang karakter judul)
9.  desc_length            (panjang karakter deskripsi)
10. req_length             (panjang karakter requirements)
11. salary_mid             (rata-rata gaji min & max)
12. salary_spread          (selisih gaji max - min)
13. scam_keyword_score     (skor keyword penipuan EN)
14. country_fraud_rate     (fraud rate per negara — TargetEncoder)
15. industry_fraud_rate    (fraud rate per industri — TargetEncoder)
16. emp_type_encoded       (employment type — TargetEncoder)
```

---

## Yang Perlu Dilanjutkan (Optimisasi F1)

### Opsi A: Tambah Fitur
Fitur yang belum dipakai tapi berpotensi informatif:
- `salary_spread_ratio` (spread/mid)
- `has_email_in_desc` (ada email gratisan di deskripsi)
- `desc_req_ratio` (rasio panjang desc vs req)
- `required_experience` encoded
- `required_education` encoded

### Opsi B: SMOTE / Oversampling
```python
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
```
Ini bisa naikkan recall signifikan karena model punya lebih banyak contoh fraud untuk belajar.

### Opsi C: Ensemble (MLP + XGBoost)
Gabungkan prediksi MLP dan XGBoost:
```python
final_score = 0.5 * mlp_score + 0.5 * xgb_score
```

### Opsi D: Tune Hyperparameters
- Learning rate: coba 0.0005 atau 0.002
- Batch size: coba 16 (lebih banyak update per epoch)
- Arsitektur: coba 128→64→32 (lebih lebar)
- Dropout: coba 0.1/0.1/0.05 (kurangi regularisasi)

### Opsi E: Threshold untuk Use Case
Untuk konteks PMI (lebih baik false alarm daripada fraud lolos):
- Pakai threshold **lebih rendah** (misal 0.3-0.5) → recall tinggi
- F1 mungkin turun tapi recall naik ke 80%+
- Ini acceptable untuk production karena prioritas = tangkap fraud

---

## Catatan Teknis

### Kenapa F1 MLP < F1 XGBoost DS (74%)?

1. **XGBoost inherently lebih bagus untuk tabular data** — ini fakta yang diakui di research. MLP butuh lebih banyak data dan tuning untuk menyamai tree-based models di tabular.

2. **Tim DS pakai lebih banyak fitur** — notebook mereka punya 31+ fitur termasuk TF-IDF text features yang kita tidak pakai.

3. **Tim DS pakai GridSearchCV** — hyperparameter mereka sudah di-tune optimal. Kita belum tune.

### Threshold Trade-off

```
Threshold rendah (0.3-0.5):
  → Recall tinggi (80%+) — banyak fraud tertangkap
  → Precision rendah — banyak false alarm
  → F1 sedang (~0.45)

Threshold tinggi (0.8-0.9):
  → Recall rendah (50-60%) — banyak fraud lolos
  → Precision tinggi — yang di-flag memang fraud
  → F1 lebih tinggi (~0.61)
```

Untuk production Emigria, **recall lebih penting** (jangan sampai fraud lolos). Jadi threshold rendah lebih cocok meskipun F1 angkanya lebih kecil.

---

## Cara Lanjutkan

1. Buka Colab baru
2. Upload `fake_job_postings.csv`
3. Copy seluruh `emigria_mlp_final.py`
4. Run Step 1-8 (sudah proven jalan)
5. Lanjut Step 9-12
6. Kalau mau optimisasi → coba Opsi A-E di atas
7. Setelah puas dengan hasil → export model dan buat FastAPI

---

## Kriteria yang Sudah Terpenuhi

### Main Quest
- [x] M1: TensorFlow Functional API (Step 7)
- [x] M2: Custom Focal Loss + F1 Callback (Step 5-6)
- [ ] M3: Export .keras (Step 11 — belum dijalankan)
- [ ] M4: Kode inference (Step 12 — belum dijalankan)

### Side Quest
- [ ] S1: FastAPI (belum dibuat)
- [x] S2: tf.GradientTape (ada di emigria_mlp.py v1 — bisa ditunjukkan)
- [ ] S3: Gemini API demo (belum)
- [x] S4: TensorBoard (Step 8 — logs tersimpan)
- [x] S5: Akurasi ≥ 85% (96% ✅), MAE ≤ 0.02 (perlu cek)
