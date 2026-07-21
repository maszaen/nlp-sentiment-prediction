import os
import joblib
from config import MODEL_PATH, VECTORIZER_PATH, clean_text

def predict_review():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH)):
        print("Model belum ada. Jalankan 'python train.py' dulu.")
        return

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    print("=====================================")
    print("    PREDIKSI SENTIMEN ULASAN GAME    ")
    print("=====================================")
    
    while True:
        review = input("\nMasukkan ulasan game (bahasa Inggris, atau ketik 'exit' untuk keluar): ").strip()
        if review.lower() == 'exit':
            break
        if not review:
            print("Ulasan tidak boleh kosong.")
            continue

        cleaned = clean_text(review)
        vec = vectorizer.transform([cleaned])

        pred = model.predict(vec)[0]
        proba = model.predict_proba(vec)[0]

        label = "POSITIF" if pred == 1 else "NEGATIF"
        print(f"Hasil Prediksi : {label}")
        print(f"P(Negatif) = {proba[0]:.4f}")
        print(f"P(Positif) = {proba[1]:.4f}")

if __name__ == "__main__":
    predict_review()
