import os
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import MODEL_PATH, VECTORIZER_PATH, TESTSET_PATH, CM_IMAGE_PATH

def evaluate_model():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(TESTSET_PATH)):
        print("Model belum ada. Jalankan 'python train.py' dulu.")
        return

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    X_test, y_test = joblib.load(TESTSET_PATH)

    X_test_tfidf = vectorizer.transform(X_test)
    preds = model.predict(X_test_tfidf)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    print("\n=== HASIL EVALUASI MODEL ===")
    print(f"Akurasi   : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, preds, target_names=["Negatif", "Positif"]))

    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Negatif", "Positif"])
    ax.set_yticklabels(["Negatif", "Positif"])
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Aktual")
    ax.set_title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(CM_IMAGE_PATH)
    plt.close(fig)
    print(f"\nGrafik confusion matrix disimpan di: {CM_IMAGE_PATH}")

if __name__ == "__main__":
    evaluate_model()
