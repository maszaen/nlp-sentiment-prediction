# Changelog

Semua perubahan penting pada proyek ini didokumentasikan di sini.
Format mengikuti prinsip [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
dan versioning menggunakan [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [v1.1.0] — 2026-07-22

### Added
- **VADER Ensemble** (`predict.py`): Menambahkan `vaderSentiment` sebagai sinyal
  kedua di luar model utama. VADER adalah lexicon-based analyzer yang paham
  degree modifier, tanda seru, dan kata negasi secara kontekstual.
- **Contrast-aware prediction** (`predict.py`, `config.py`): Saat kata seperti
  `"but"`, `"however"`, `"overall"`, `"thankfully"` terdeteksi, kalimat terakhir
  diberi bobot lebih besar (60%) karena biasanya berisi verdict akhir reviewer.
- **Helper functions** (`config.py`): `get_last_sentence()`, `detect_contrast()`,
  `has_positive_conclusion()` — fungsi utilitas untuk analisis struktur review.
- **Konstanta baru** (`config.py`): `CONTRAST_WORDS`, `POSITIVE_CONCLUSION_WORDS`.
- **Info ensemble di output** (`predict.py`): Output kini menampilkan skor model
  dan VADER secara terpisah, dan menandai jika mode contrast aktif.

### Changed
- **Algoritma model** (`train.py`): Upgrade dari `MultinomialNB` ke `LinearSVC`
  dengan `CalibratedClassifierCV` untuk menghasilkan probabilitas yang terkalibrasi.
  LinearSVC lebih akurat karena tidak mengasumsikan independensi antar kata.
- **Penanganan class imbalance** (`train.py`): Parameter `class_weight='balanced'`
  pada `LinearSVC` mengoreksi bias akibat distribusi data yang tidak seimbang
  (81.8% positif vs 18.2% negatif pada dataset Steam reviews).
- **Bobot ensemble** (`predict.py`): Model utama 65%, VADER 35% (dapat dituning).

### Fixed
- Kasus review dengan pola *"negatif di awal, positif di akhir"* yang sebelumnya
  salah diprediksi karena kata-kata kuat negatif (`"insufferable"`, `"ruined"`)
  mendominasi bag-of-words. Kini kalimat terakhir dianalisis secara terpisah.

---

## [v1.0.0] — 2026-07-21

### Initial Release

Implementasi awal sistem prediksi sentimen ulasan game berbasis NLP.

#### Arsitektur
- **Dataset**: Steam Game Reviews (`steam_game_reviews_730945.csv`)
  — 730.945 ulasan, distribusi 81.8% positif / 18.2% negatif
- **Preprocessing** (`config.py`): Lowercase, hapus URL, tanda baca, angka,
  stopword; kata negasi (`not`, `no`, `never`, dll.) dipertahankan secara eksplisit
- **Vectorizer**: TF-IDF dengan 50.000 fitur maksimum, n-gram (1, 3)
- **Model**: `MultinomialNB` dengan `fit_prior=False`
- **Split data**: 80% train / 20% test, stratified

#### File
| File | Deskripsi |
|---|---|
| `config.py` | Konfigurasi path, stopword, fungsi `clean_text()` |
| `train.py` | Pipeline training: load → preprocess → vectorize → fit → save |
| `predict.py` | Antarmuka CLI prediksi interaktif |
| `evaluate.py` | Evaluasi model: accuracy, precision, recall, F1, confusion matrix |

#### Performa (v1.0)
- **Akurasi**: ~84–86% (dari confusion matrix)
- **Kelebihan**: Cepat dilatih, ringan, interpretable
- **Kekurangan**:
  - Tidak menangani sarkasme
  - Tidak paham konteks (word order diabaikan)
  - Kata kuat negatif sering mendominasi pada review campuran
  - Class imbalance tidak dikoreksi secara eksplisit
