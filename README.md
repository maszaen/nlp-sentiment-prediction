# 🎮 NLP Sentiment Analysis — Steam Game Reviews

Sistem analisis sentimen otomatis untuk ulasan game Steam menggunakan **Machine Learning** dan **VADER Ensemble**. Proyek ini dibangun untuk UAS mata kuliah Kecerdasan Buatan.

---

## 🧠 Arsitektur Sistem (v1.1)

```
Teks Ulasan (raw)
        │
        ▼
┌───────────────────┐
│   Preprocessing   │  clean_text(): lowercase, hapus URL/angka/
│   (config.py)     │  tanda baca, filter stopword, pertahankan negasi
└────────┬──────────┘
         │
         ▼
┌────────────────────┐     ┌─────────────────────────┐
│  TF-IDF Vectorizer │     │  VADER Sentiment Analyzer│
│  50.000 fitur      │     │  (lexicon-based, real-time│
│  n-gram (1,2,3)    │     │   tanpa training)        │
└────────┬───────────┘     └────────────┬────────────┘
         │                              │
         ▼                              ▼
┌────────────────────┐     ┌─────────────────────────┐
│  LinearSVC Model   │     │  Contrast Detection      │
│  class_weight=     │     │  - Full-text score       │
│  'balanced'        │     │  - Last-sentence score   │
│  + Kalibasi Proba  │     │  - Contrast-mode weighting│
└────────┬───────────┘     └────────────┬────────────┘
         │                              │
         └──────────────┬───────────────┘
                        ▼
              ┌─────────────────┐
              │ Ensemble Score  │
              │ 65% Model       │
              │ 35% VADER       │
              └────────┬────────┘
                       ▼
              ┌─────────────────┐
              │ POSITIF/NEGATIF │
              │ + Probabilitas  │
              └─────────────────┘
```

---

## 📦 Dataset

| Atribut | Detail |
|---|---|
| Sumber | Steam Game Reviews |
| File | `steam_game_reviews_730945.csv` |
| Total data | 730.945 ulasan |
| Distribusi | 81.8% Positif / 18.2% Negatif |
| Fitur utama | `review` (teks), `voted_up` (label boolean) |
| Rata-rata panjang | ~318 karakter per ulasan |

> **Catatan**: File dataset tidak disertakan di repo karena ukurannya ~285MB.
> Unduh dan letakkan di root folder dengan nama `steam_game_reviews_730945.csv`.

---

## 🚀 Quick Start

### 1. Prerequisites

```bash
pip install scikit-learn pandas joblib vaderSentiment matplotlib
```

### 2. Struktur Folder

```
.
├── config.py          # Preprocessing, konstanta, helper functions
├── train.py           # Pipeline pelatihan model
├── evaluate.py        # Evaluasi model (akurasi, F1, confusion matrix)
├── predict.py         # CLI prediksi interaktif (dengan VADER ensemble)
├── CHANGELOG.md       # Log perubahan per versi
├── model/
│   ├── nb_model.pkl          # Model terlatih
│   ├── tfidf_vectorizer.pkl  # TF-IDF vectorizer terlatih
│   ├── test_split.pkl        # Data test tersimpan
│   └── confusion_matrix.png  # Visualisasi confusion matrix
└── steam_game_reviews_730945.csv   # Dataset (tidak disertakan di repo)
```

### 3. Melatih Model

```bash
python train.py
```

Model dilatih menggunakan **LinearSVC** (`class_weight='balanced'`) dengan TF-IDF 50K fitur dan n-gram (1–3).
Proses training memerlukan waktu sekitar 30–60 menit tergantung hardware.

### 4. Evaluasi Model

```bash
python evaluate.py
```

Menampilkan: Akurasi, Precision, Recall, F1-Score, Classification Report, dan menyimpan Confusion Matrix ke `model/confusion_matrix.png`.

### 5. Prediksi Interaktif

```bash
python predict.py
```

```
=====================================
    PREDIKSI SENTIMEN ULASAN GAME
=====================================

Masukkan ulasan game (bahasa Inggris, atau ketik 'exit' untuk keluar): The combat system is so satisfying!
Hasil Prediksi : POSITIF
P(Negatif)     = 0.0521
P(Positif)     = 0.9479
  [Model=0.9295 | VADER=0.9612]
```

---

## ⚙️ Konfigurasi Ensemble

Bobot ensemble dapat dituning langsung di `predict.py`:

```python
MODEL_WEIGHT    = 0.65   # Porsi model utama (TF-IDF + LinearSVC)
VADER_WEIGHT    = 0.35   # Porsi VADER (lexicon-based)
LAST_SENT_BOOST = 0.60   # Bobot kalimat terakhir saat contrast terdeteksi
FULL_TEXT_RATIO = 0.40   # Bobot full-text VADER saat contrast terdeteksi
```

---

## 📊 Performa Model

### v1.0 (MultinomialNB)
| Metrik | Nilai |
|---|---|
| Akurasi | ~84–86% |
| Kelemahan | Bias ke Positif (class imbalance), gagal pada review campuran |

### v1.1 (LinearSVC + VADER Ensemble)
| Metrik | Nilai |
|---|---|
| Akurasi (estimasi) | ~88–91% |
| Perbaikan | Class imbalance fix, konteks kalimat terakhir, contrast detection |

---

## 🔍 Fitur Utama

- ✅ **VADER Ensemble**: Kombinasi model statistik + lexicon-based untuk konteks yang lebih baik
- ✅ **Contrast-aware**: Mendeteksi pola "negatif di awal, positif di akhir" (*"...but I still enjoyed it overall"*)
- ✅ **Last-sentence weighting**: Kalimat penutup review diberi bobot lebih besar saat contrast terdeteksi
- ✅ **Class imbalance handling**: `class_weight='balanced'` pada LinearSVC
- ✅ **Negation preservation**: Kata `not`, `no`, `never`, dll. tidak dihapus dari stopword

---

## 📝 Changelog

Lihat [CHANGELOG.md](CHANGELOG.md) untuk detail perubahan per versi.

---

## 🎓 Konteks Proyek

Proyek ini dibuat sebagai tugas UAS mata kuliah **Kecerdasan Buatan** pada domain **Natural Language Processing (NLP)** — analisis sentimen ulasan game Steam.
