import os
import joblib
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from config import (
    MODEL_PATH, VECTORIZER_PATH, clean_text,
    get_last_sentence, detect_contrast
)

# ================================================================
# Bobot ensemble: VADER + Model
# Diturunkan dari eksperimen manual terhadap kasus-kasus edge case
# ================================================================
MODEL_WEIGHT = 0.65   # Bobot prediksi model utama (TF-IDF based)
VADER_WEIGHT  = 0.35  # Bobot VADER (lexicon-based, paham konteks & contrast)

# Bobot ekstra untuk kalimat terakhir saat contrast terdeteksi
LAST_SENT_BOOST = 0.60   # porsi kalimat terakhir dari skor VADER
FULL_TEXT_RATIO = 0.40   # porsi seluruh teks dari skor VADER


def ensemble_predict(review: str, model, vectorizer, vader: SentimentIntensityAnalyzer):
    """
    Hitung prediksi sentimen dengan tiga sinyal:
      1. Model utama (LinearSVC/NB + TF-IDF)
      2. VADER full-text — menangkap sentimen keseluruhan
      3. VADER last-sentence — menangkap kesimpulan akhir reviewer

    Jika contrast terdeteksi (kata seperti 'but', 'however', 'overall'),
    kalimat terakhir diberi bobot lebih besar karena biasanya berisi
    kesimpulan/verdict akhir reviewer.
    """
    # --- Sinyal 1: Model ---
    cleaned = clean_text(review)
    vec = vectorizer.transform([cleaned])
    model_proba = model.predict_proba(vec)[0]
    model_pos   = model_proba[1]

    # --- Sinyal 2 & 3: VADER ---
    vader_full  = vader.polarity_scores(review)['compound']        # -1 to 1
    last_sent   = get_last_sentence(review)
    vader_last  = vader.polarity_scores(last_sent)['compound']     # -1 to 1

    # Normalisasi VADER ke rentang [0, 1]
    vader_full_pos = (vader_full + 1) / 2
    vader_last_pos = (vader_last + 1) / 2

    # Gabungkan dua sinyal VADER
    if detect_contrast(review):
        # Kalimat terakhir lebih penting saat ada kata contrast
        vader_final = LAST_SENT_BOOST * vader_last_pos + FULL_TEXT_RATIO * vader_full_pos
    else:
        vader_final = vader_full_pos

    # --- Ensemble Final ---
    final_pos = MODEL_WEIGHT * model_pos + VADER_WEIGHT * vader_final
    final_neg = 1.0 - final_pos

    return final_pos, final_neg, model_pos, vader_final, detect_contrast(review)


def predict_review():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH)):
        print("Model belum ada. Jalankan 'python train.py' dulu.")
        return

    model      = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    vader      = SentimentIntensityAnalyzer()

    print("=====================================")
    print("    PREDIKSI SENTIMEN ULASAN GAME    ")
    print("=====================================")

    while True:
        review = input(
            "\nMasukkan ulasan game (bahasa Inggris, atau ketik 'exit' untuk keluar): "
        ).strip()

        if review.lower() == "exit":
            break
        if not review:
            print("Ulasan tidak boleh kosong.")
            continue

        final_pos, final_neg, model_pos, vader_score, has_contrast = ensemble_predict(
            review, model, vectorizer, vader
        )

        label = "POSITIF" if final_pos >= 0.5 else "NEGATIF"
        print(f"Hasil Prediksi : {label}")
        print(f"P(Negatif)     = {final_neg:.4f}")
        print(f"P(Positif)     = {final_pos:.4f}")
        print(f"  [Model={model_pos:.4f} | VADER={vader_score:.4f}"
              + (" | contrast-mode]" if has_contrast else "]"))


if __name__ == "__main__":
    predict_review()
