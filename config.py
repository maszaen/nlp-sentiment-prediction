import os
import re
import string
from sklearn.feature_extraction import text as sk_text

DATA_PATH = "steam_game_reviews_730945.csv"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "nb_model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
TESTSET_PATH = os.path.join(MODEL_DIR, "test_split.pkl")
CM_IMAGE_PATH = os.path.join(MODEL_DIR, "confusion_matrix.png")

# Stopword list bawaan sklearn dipakai, tapi kita hilangkan kata negasi
# agar kata seperti 'not', 'no' tidak terhapus (penting untuk sentimen)
base_stopwords = set(sk_text.ENGLISH_STOP_WORDS)
negation_words = {
    "not", "no", "nor", "none", "neither", "never", "cannot",
    "dont", "don", "cant", "can", "didn", "didnt", "isn", "isnt",
    "aren", "arent", "wasn", "wasnt", "weren", "werent", "won", "wont"
}
CUSTOM_STOPWORDS = base_stopwords - negation_words

def clean_text(text: str) -> str:
    """
    Langkah preprocessing:
    1. Lowercase semua huruf
    2. Hapus angka dan tanda baca
    3. Hapus spasi berlebih
    4. Hapus stopword (kata umum yang tidak bermakna sentimen)
    """
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)          # hapus URL
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)  # hapus tanda baca
    text = re.sub(r"\d+", " ", text)                      # hapus angka
    text = re.sub(r"\s+", " ", text).strip()              # rapikan spasi

    tokens = [w for w in text.split() if w not in CUSTOM_STOPWORDS and len(w) > 1]
    return " ".join(tokens)
