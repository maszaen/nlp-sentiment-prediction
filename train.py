import os
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from config import DATA_PATH, MODEL_DIR, MODEL_PATH, VECTORIZER_PATH, TESTSET_PATH, clean_text

def train_model():
    print("Memuat dataset...")
    if not os.path.exists(DATA_PATH):
        print(f"Dataset {DATA_PATH} tidak ditemukan.")
        return

    df = pd.read_csv(DATA_PATH)
    
    # Gunakan FULL DATA 100% tanpa di-sample
    df = df[["review", "voted_up"]].dropna()
    df["label"] = df["voted_up"].apply(lambda x: 1 if x else 0)

    print("Membersihkan teks ulasan (preprocessing)...")
    df["clean_review"] = df["review"].apply(clean_text)
    df = df[df["clean_review"].str.len() > 0]

    X = df["clean_review"]
    y = df["label"]

    print(f"Total data setelah preprocessing: {len(df)} ulasan")
    print(f"Distribusi label -> Positif: {(y == 1).sum()}, Negatif: {(y == 0).sum()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Tambah kosa kata ke 50.000 dan ambil hingga 3-gram
    vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 3))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print("Melatih model Multinomial Naive Bayes...")
    # fit_prior=False memaksa model tidak terpengaruh oleh dominasi jumlah Positif
    model = MultinomialNB(fit_prior=False)
    model.fit(X_train_tfidf, y_train)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump((X_test, y_test), TESTSET_PATH)

    preds = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, preds)
    print(f"\nTraining selesai. Akurasi pada data uji: {acc:.4f}")
    print(f"Model tersimpan di: {MODEL_PATH}")

if __name__ == "__main__":
    train_model()
